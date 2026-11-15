from __future__ import annotations

from collections.abc import Iterable
from typing import Type

import torch
import torch.distributed as dist



def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    from systems.flash_attention import FlashAttentionPytorchFunction

    return FlashAttentionPytorchFunction


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    from systems.flash_attention import FlashAttentionTritonFunction

    return FlashAttentionTritonFunction


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return _IndividualParameterDDP(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float | None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    return _BucketedDDP(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.start_gradient_synchronization()


def get_sharded_optimizer(
    params: Iterable[torch.Tensor] | Iterable[dict],
    optimizer_cls: Type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return _ShardedOptimizer(params, optimizer_cls, **kwargs)


def _require_process_group() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("A torch.distributed process group must be initialized")
    return dist.get_rank(), dist.get_world_size()


class _DistributedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        _require_process_group()
        self.module = module
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _IndividualParameterDDP(_DistributedModule):
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self._pending_reductions = []
        self._gradient_hooks = [
            parameter.register_post_accumulate_grad_hook(self._make_hook(parameter))
            for parameter in self.module.parameters()
            if parameter.requires_grad
        ]

    def _make_hook(self, parameter: torch.nn.Parameter):
        def synchronize_gradient(_):
            gradient = parameter.grad
            if gradient is not None:
                work = dist.all_reduce(gradient, async_op=True)
                self._pending_reductions.append((work, gradient))

        return synchronize_gradient

    def finish_gradient_synchronization(self):
        _, world_size = _require_process_group()
        pending, self._pending_reductions = self._pending_reductions, []
        for work, gradient in pending:
            work.wait()
            gradient.div_(world_size)


class _BucketedDDP(_DistributedModule):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None):
        super().__init__(module)
        if bucket_size_mb is not None and bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be positive or None")

        self._buckets = self._make_buckets(bucket_size_mb)
        self._bucket_by_parameter = {
            parameter: bucket_index
            for bucket_index, bucket in enumerate(self._buckets)
            for parameter in bucket
        }
        self._ready_parameters: set[torch.nn.Parameter] = set()
        self._pending_reductions = []
        self._gradient_hooks = [
            parameter.register_post_accumulate_grad_hook(self._make_hook())
            for parameter in self.module.parameters()
            if parameter.requires_grad
        ]

    def _make_buckets(self, bucket_size_mb: float | None) -> list[list[torch.nn.Parameter]]:
        parameters = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        if not parameters:
            return []

        if bucket_size_mb is None:
            return [parameters]

        max_bucket_bytes = int(bucket_size_mb * 1024 * 1024)
        buckets = []
        current_bucket = []
        current_bytes = 0
        current_device = None
        current_dtype = None
        for parameter in parameters:
            parameter_bytes = parameter.numel() * parameter.element_size()
            if current_bucket and (
                parameter.device != current_device
                or parameter.dtype != current_dtype
                or current_bytes >= max_bucket_bytes
            ):
                buckets.append(current_bucket)
                current_bucket = []
                current_bytes = 0

            current_bucket.append(parameter)
            current_bytes += parameter_bytes
            current_device = parameter.device
            current_dtype = parameter.dtype

        if current_bucket:
            buckets.append(current_bucket)
        return buckets

    def _make_hook(self):
        def mark_gradient_ready(parameter):
            self._ready_parameters.add(parameter)
            bucket_index = self._bucket_by_parameter.get(parameter)
            if bucket_index is None:
                return
            bucket = self._buckets[bucket_index]
            if all(item in self._ready_parameters for item in bucket):
                self._reduce_bucket(bucket)

        return mark_gradient_ready

    def _reduce_bucket(self, bucket: list[torch.nn.Parameter]):
        gradients = [parameter.grad for parameter in bucket]
        if any(gradient is None for gradient in gradients):
            return
        flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
        work = dist.all_reduce(flat_gradient, async_op=True)
        self._pending_reductions.append((work, flat_gradient, bucket))

    def start_gradient_synchronization(self):
        if self._pending_reductions:
            raise RuntimeError("Previous gradient synchronization has not finished")
        self._ready_parameters.clear()

    def finish_gradient_synchronization(self):
        _, world_size = _require_process_group()
        pending, self._pending_reductions = self._pending_reductions, []
        for work, flat_gradient, bucket in pending:
            work.wait()
            flat_gradient.div_(world_size)
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad.copy_(flat_gradient[offset : offset + count].view_as(parameter))
                offset += count
        self._ready_parameters.clear()


class _ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        rank, world_size = _require_process_group()
        super().__init__(params, defaults=kwargs)

        local_groups = []
        owner_by_parameter = {}
        parameter_index = 0
        for group in self.param_groups:
            local_parameters = []
            for parameter in group["params"]:
                owner = parameter_index % world_size
                owner_by_parameter[parameter] = owner
                if owner == rank:
                    local_parameters.append(parameter)
                parameter_index += 1
            if local_parameters:
                local_groups.append({**group, "params": local_parameters})

        self._all_parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        self._owner_by_parameter = owner_by_parameter
        self._local_optimizer = optimizer_cls(local_groups) if local_groups else None
        if self._local_optimizer is not None:
            self.state = self._local_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        return super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure)
        else:
            loss = closure() if closure is not None else None

        with torch.no_grad():
            for parameter in self._all_parameters:
                dist.broadcast(parameter, src=self._owner_by_parameter[parameter])
        return loss

    def state_dict(self):
        if self._local_optimizer is None:
            return {"state": {}, "param_groups": []}
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if self._local_optimizer is None:
            if state_dict["state"] or state_dict["param_groups"]:
                raise ValueError("This rank owns no optimizer parameters")
            return
        self._local_optimizer.load_state_dict(state_dict)

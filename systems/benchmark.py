"""End-to-end benchmarking + profiling harness for the basics Transformer LM.

Covers: benchmarking_script, nsys_profile (NVTX ranges), benchmarking_mixed_precision,
memory_profiling and torch_compile (b) of the handout.

Examples
--------
    # forward + backward timing, small model, 5 warm-up / 10 measured steps
    uv run python -m systems.benchmark --size small --context-length 512 --mode forward_backward

    # full training step with BF16 autocast
    uv run python -m systems.benchmark --size medium --mode train --precision bf16

    # no warm-up (benchmarking_script (c))
    uv run python -m systems.benchmark --size small --warmup-steps 0

    # nsys (warm-up steps are tagged "warmup_*", measured steps "measure_*")
    uv run nsys profile -o small_512 --python-backtrace=cuda python -m systems.benchmark \
        --size small --context-length 512 --mode train --nvtx

    # memory profile (writes a snapshot you can drop on https://pytorch.org/memory_viz)
    uv run python -m systems.benchmark --size small --context-length 512 --mode train \
        --memory-snapshot small_512_train.pickle

    # torch.compile'd model
    uv run python -m systems.benchmark --size small --compile
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import statistics
import timeit
from typing import Any

import torch

import basics.model
from basics.model import BasicsTransformerLM
from basics.nn_utils import cross_entropy, softmax
from basics.optimizer import AdamW

# Table 1 of the handout.
MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
ROPE_THETA = 10_000.0

MODES = ("forward", "forward_backward", "train")
PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------
def nvtx_range(name: str, enabled: bool):
    """NVTX range context manager; a no-op when disabled (or on CPU-only machines)."""
    if enabled:
        import torch.cuda.nvtx as nvtx

        return nvtx.range(name)
    return contextlib.nullcontext()


def install_annotated_attention() -> None:
    """Swap basics' attention for a version with NVTX ranges around each stage."""
    import torch.cuda.nvtx as nvtx
    from einops import einsum

    @nvtx.range("scaled dot product attention")
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        with nvtx.range("computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
        return out

    # CausalMultiHeadSelfAttention looks the function up as a module global in basics.model.
    basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build_model(size: str, context_length: int, device: str) -> BasicsTransformerLM:
    cfg = MODEL_SIZES[size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        rope_theta=ROPE_THETA,
        **cfg,
    )
    return model.to(device)


def run_benchmark(
    size: str = "small",
    context_length: int = 512,
    mode: str = "forward_backward",
    warmup_steps: int = 5,
    steps: int = 10,
    precision: str = "fp32",
    compile_model: bool = False,
    device: str | None = None,
    nvtx: bool = False,
    memory_snapshot: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Time `steps` iterations after `warmup_steps` warm-up iterations.

    Returns a dict with mean/std (seconds) for the forward pass and, depending on `mode`, the backward pass
    and optimizer step, plus the peak CUDA memory (MiB) over the measured steps.
    """
    assert mode in MODES, f"mode must be one of {MODES}"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.startswith("cuda")
    if nvtx and not use_cuda:
        raise ValueError("--nvtx requires CUDA")
    if nvtx:
        install_annotated_attention()

    torch.manual_seed(seed)
    model = build_model(size, context_length, device)
    optimizer = AdamW(model.parameters(), lr=1e-4) if mode == "train" else None
    if compile_model:
        model = torch.compile(model)

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)

    amp_dtype = PRECISIONS[precision]
    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
        if amp_dtype is not None
        else contextlib.nullcontext
    )

    fwd_times: list[float] = []
    bwd_times: list[float] = []
    opt_times: list[float] = []

    def one_step(record: bool) -> None:
        # ---- forward (+ loss) ----
        t0 = timeit.default_timer()
        with nvtx_range("forward", nvtx):
            if mode == "forward":
                with torch.no_grad(), autocast_ctx():
                    model(x)
            else:
                with autocast_ctx():
                    logits = model(x)
                    loss = cross_entropy(logits.float(), y)
            sync(device)
        t1 = timeit.default_timer()

        # ---- backward ----
        if mode != "forward":
            with nvtx_range("backward", nvtx):
                loss.backward()
                sync(device)
        t2 = timeit.default_timer()

        # ---- optimizer ----
        if mode == "train":
            with nvtx_range("optimizer step", nvtx):
                optimizer.step()
                sync(device)
        t3 = timeit.default_timer()

        # Clear grads outside of the timed regions.
        if mode != "forward":
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

        if record:
            fwd_times.append(t1 - t0)
            if mode != "forward":
                bwd_times.append(t2 - t1)
            if mode == "train":
                opt_times.append(t3 - t2)

    # ---- warm-up ----
    for i in range(warmup_steps):
        with nvtx_range(f"warmup_{i}", nvtx):
            one_step(record=False)
    sync(device)

    # ---- measurement ----
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    if memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    for i in range(steps):
        with nvtx_range(f"measure_{i}", nvtx):
            one_step(record=True)
    sync(device)

    if memory_snapshot:
        torch.cuda.memory._dump_snapshot(memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)

    def ms(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return (
            statistics.fmean(values) * 1e3,
            (statistics.stdev(values) * 1e3) if len(values) > 1 else 0.0,
        )

    fwd_mean, fwd_std = ms(fwd_times)
    bwd_mean, bwd_std = ms(bwd_times)
    opt_mean, opt_std = ms(opt_times)
    result: dict[str, Any] = dict(
        size=size,
        context_length=context_length,
        mode=mode,
        precision=precision,
        compiled=compile_model,
        warmup_steps=warmup_steps,
        steps=steps,
        forward_ms_mean=fwd_mean,
        forward_ms_std=fwd_std,
        backward_ms_mean=bwd_mean,
        backward_ms_std=bwd_std,
        optimizer_ms_mean=opt_mean,
        optimizer_ms_std=opt_std,
        peak_memory_mib=(torch.cuda.max_memory_allocated() / 2**20) if use_cuda else None,
    )

    # Free GPU memory so sweeps can reuse the device.
    del model, optimizer, x, y
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=MODEL_SIZES.keys(), default="small")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--mode", choices=MODES, default="forward_backward")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--precision", choices=PRECISIONS.keys(), default="fp32")
    p.add_argument("--compile", action="store_true", help="wrap the model with torch.compile")
    p.add_argument("--device", default=None)
    p.add_argument("--nvtx", action="store_true", help="add NVTX ranges (for nsys) incl. annotated attention")
    p.add_argument("--memory-snapshot", default=None, help="path for torch memory snapshot pickle")
    p.add_argument("--output", default=None, help="append the result as a JSON line to this file")
    args = p.parse_args()

    try:
        result = run_benchmark(
            size=args.size,
            context_length=args.context_length,
            mode=args.mode,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            precision=args.precision,
            compile_model=args.compile,
            device=args.device,
            nvtx=args.nvtx,
            memory_snapshot=args.memory_snapshot,
        )
    except torch.cuda.OutOfMemoryError:
        result = dict(
            size=args.size, context_length=args.context_length, mode=args.mode, precision=args.precision, error="OOM"
        )

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

"""FlashAttention-2 implementations.

Contents
--------
* ``FlashAttentionPytorchFunction``: tiled FA-2 forward written in plain PyTorch
  (Algorithm 1). Slow, but useful as a reference when debugging the Triton kernel.
* ``flash_fwd_kernel`` / ``FlashAttentionTritonFunction``: fused Triton forward kernel
  (with optional causal masking) wrapped in an ``autograd.Function``.
* Both classes share the same backward pass: recomputation using the saved
  logsumexp ``L`` (Eq. 13-19), written in plain PyTorch and wrapped in ``torch.compile``.
"""

from __future__ import annotations

import math

import torch

try:  # Triton is only needed for the GPU kernel; keep the PyTorch version importable without it.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Backward pass (shared): recomputation with L, no softmax needed.
# ---------------------------------------------------------------------------
def _flash_backward_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
):
    """Eq. 13-19 of the handout.

    Shapes: Q, O, dO: (B, Nq, d); K, V: (B, Nk, d); L: (B, Nq).
    Returns dQ, dK, dV.
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    n_queries, n_keys = Q.shape[-2], K.shape[-2]

    # Do the math in fp32 regardless of the input dtype.
    Qf, Kf, Vf, Of, dOf = (t.float() for t in (Q, K, V, O, dO))
    Lf = L.float()

    # Precompute D = rowsum(O * dO)  (equal to rowsum(P * dP)).
    D = (Of * dOf).sum(dim=-1, keepdim=True)  # (B, Nq, 1)

    # (13) S = Q K^T / sqrt(d)
    S = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale  # (B, Nq, Nk)
    if is_causal:
        q_idx = torch.arange(n_queries, device=Q.device)[:, None]
        k_idx = torch.arange(n_keys, device=Q.device)[None, :]
        S = torch.where(q_idx >= k_idx, S, S - 1e6)

    # (14) P = exp(S - L)
    P = torch.exp(S - Lf.unsqueeze(-1))
    # (15) dV = P^T dO
    dV = torch.matmul(P.transpose(-1, -2), dOf)
    # (16) dP = dO V^T
    dP = torch.matmul(dOf, Vf.transpose(-1, -2))
    # (17) dS = P * (dP - D)
    dS = P * (dP - D)
    # (18) dQ = dS K / sqrt(d)
    dQ = torch.matmul(dS, Kf) * scale
    # (19) dK = dS^T Q / sqrt(d)
    dK = torch.matmul(dS.transpose(-1, -2), Qf) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


_compiled_backward = None
_compile_failed = False


def _flash_backward(Q, K, V, O, dO, L, is_causal):
    """Run the backward math through torch.compile (falls back to eager if compilation is unavailable,
    e.g. on a machine without a C++ compiler for the CPU inductor backend)."""
    global _compiled_backward, _compile_failed
    if not _compile_failed:
        try:
            if _compiled_backward is None:
                _compiled_backward = torch.compile(_flash_backward_impl)
            return _compiled_backward(Q, K, V, O, dO, L, is_causal)
        except Exception:  # pragma: no cover - environment dependent
            _compile_failed = True
    return _flash_backward_impl(Q, K, V, O, dO, L, is_causal)


# ---------------------------------------------------------------------------
# (a) Pure PyTorch tiled FlashAttention-2 forward
# ---------------------------------------------------------------------------
class FlashAttentionPytorchFunction(torch.autograd.Function):
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        Bq = min(FlashAttentionPytorchFunction.Q_TILE_SIZE, Nq)
        Bk = min(FlashAttentionPytorchFunction.K_TILE_SIZE, Nk)
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        scale = 1.0 / math.sqrt(d)

        O = torch.empty_like(Q)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        for i in range(Tq):
            q_start, q_end = i * Bq, min((i + 1) * Bq, Nq)
            Qi = Q[:, q_start:q_end, :].float()  # (B, bq, d)
            bq = q_end - q_start

            Oi = torch.zeros((B, bq, d), device=Q.device, dtype=torch.float32)
            li = torch.zeros((B, bq), device=Q.device, dtype=torch.float32)
            mi = torch.full((B, bq), float("-inf"), device=Q.device, dtype=torch.float32)

            q_idx = torch.arange(q_start, q_end, device=Q.device)

            for j in range(Tk):
                k_start, k_end = j * Bk, min((j + 1) * Bk, Nk)
                Kj = K[:, k_start:k_end, :].float()
                Vj = V[:, k_start:k_end, :].float()

                Sij = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale  # (B, bq, bk)
                if is_causal:
                    k_idx = torch.arange(k_start, k_end, device=Q.device)
                    mask = q_idx[:, None] >= k_idx[None, :]
                    Sij = torch.where(mask, Sij, Sij - 1e6)

                m_new = torch.maximum(mi, Sij.max(dim=-1).values)
                P_tilde = torch.exp(Sij - m_new.unsqueeze(-1))
                alpha = torch.exp(mi - m_new)
                li = alpha * li + P_tilde.sum(dim=-1)
                Oi = alpha.unsqueeze(-1) * Oi + torch.matmul(P_tilde, Vj)
                mi = m_new

            Oi = Oi / li.unsqueeze(-1)
            Li = mi + torch.log(li)

            O[:, q_start:q_end, :] = Oi.to(O.dtype)
            L[:, q_start:q_end] = Li

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


# ---------------------------------------------------------------------------
# (b, c) Triton FlashAttention-2 forward kernel
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Block pointers, offset by the batch index.
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load the query tile once; it stays on-chip for the whole key loop.
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bq, D)

        # On-chip accumulators in fp32.
        o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        # Query indices for the causal mask.
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)

            # S_ij = Q_i K_j^T / sqrt(d)
            s = tl.dot(q, tl.trans(k)) * scale  # (Bq, Bk) fp32

            if is_causal:
                k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                mask = q_idx[:, None] >= k_idx[None, :]
                s = tl.where(mask, s, s - 1e6)

            # Online softmax update.
            m_new = tl.maximum(m, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m - m_new)
            l = alpha * l + tl.sum(p, axis=1)
            o = alpha[:, None] * o
            o = tl.dot(p.to(v.dtype), v, acc=o)
            m = m_new

            # Advance to the next key tile.
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Normalise and write back.
        o = o / l[:, None]
        lse = m + tl.log(l)

        tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, lse, boundary_check=(0,))


class FlashAttentionTritonFunction(torch.autograd.Function):
    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert HAS_TRITON, "Triton is required for FlashAttentionTritonFunction"
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3, "Expected (batch, seq, d) tensors"

        B, Nq, D = Q.shape
        Nk = K.shape[1]
        Bq = FlashAttentionTritonFunction.Q_TILE_SIZE
        Bk = FlashAttentionTritonFunction.K_TILE_SIZE

        O = torch.empty_like(Q)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(Nq, Bq), B)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=bool(is_causal),
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None

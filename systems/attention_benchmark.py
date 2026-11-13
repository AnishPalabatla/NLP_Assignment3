"""Benchmark naive PyTorch attention (and a torch.compile'd version) at different scales.

Covers `pytorch_attention` and `torch_compile (a)`.

    uv run python -m systems.attention_benchmark
    uv run python -m systems.attention_benchmark --compile --latex

Setup: batch size 8, no head dimension, d_model in [16, 32, 64, 128], seq_len in
[256, 1024, 4096, 8192, 16384]; 100 forward passes, then memory-in-use right before backward and
100 backward passes. Out-of-memory configurations are reported as OOM.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import timeit

import torch

import basics.model
from systems.tables import rows_to_latex, rows_to_markdown

BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


def benchmark_attention(attn_fn, d_model: int, seq_len: int, n_iters: int = 100, warmup: int = 10, causal: bool = False):
    device = "cuda"
    q, k, v = (torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True) for _ in range(3))
    mask = None
    if causal:
        idx = torch.arange(seq_len, device=device)
        mask = idx[:, None] >= idx[None, :]

    def fwd():
        return attn_fn(q, k, v, mask)

    # ---- warm-up (forward + backward) ----
    for _ in range(warmup):
        out = fwd()
        out.sum().backward()
    q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    # ---- forward timing ----
    fwd_times = []
    for _ in range(n_iters):
        t0 = timeit.default_timer()
        out = fwd()
        torch.cuda.synchronize()
        fwd_times.append(timeit.default_timer() - t0)

    # ---- memory before backward + backward timing ----
    # Each backward needs a fresh graph, so run an (untimed) forward before every backward.
    bwd_times = []
    mem_before_bwd = None
    for _ in range(n_iters):
        out = fwd()
        loss = out.sum()
        torch.cuda.synchronize()
        if mem_before_bwd is None:
            mem_before_bwd = torch.cuda.memory_allocated() / 2**20
        t0 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append(timeit.default_timer() - t0)
        del out, loss
        q.grad = k.grad = v.grad = None

    return dict(
        forward_ms=statistics.fmean(fwd_times) * 1e3,
        backward_ms=statistics.fmean(bwd_times) * 1e3,
        mem_before_bwd_mib=mem_before_bwd,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compile", action="store_true", help="also benchmark torch.compile'd attention")
    p.add_argument("--causal", action="store_true", help="apply a causal mask")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()

    impls = {"eager": basics.model.scaled_dot_product_attention}
    if args.compile:
        impls["compiled"] = torch.compile(basics.model.scaled_dot_product_attention)

    rows = []
    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        row = dict(d_model=d_model, seq_len=seq_len)
        for name, fn in impls.items():
            print(f"--> {name} d_model={d_model} seq_len={seq_len}", flush=True)
            try:
                res = benchmark_attention(fn, d_model, seq_len, n_iters=args.iters, causal=args.causal)
                row[f"{name}_fwd_ms"] = res["forward_ms"]
                row[f"{name}_bwd_ms"] = res["backward_ms"]
                row[f"{name}_mem_before_bwd_mib"] = res["mem_before_bwd_mib"]
            except torch.cuda.OutOfMemoryError:
                row[f"{name}_fwd_ms"] = row[f"{name}_bwd_ms"] = "OOM"
                row[f"{name}_mem_before_bwd_mib"] = "OOM"
            finally:
                torch.cuda.empty_cache()
        rows.append(row)

    print()
    print(rows_to_markdown(rows))
    if args.latex:
        print()
        print(rows_to_latex(rows))


if __name__ == "__main__":
    main()

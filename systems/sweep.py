"""Sweep over benchmark configurations and print markdown / LaTeX tables for the writeup.

Examples
--------
    # benchmarking_script (b): forward + backward timings, both sizes, 5 warm-up / 10 steps
    uv run python -m systems.sweep --sizes small medium --context-lengths 128 256 512 1024

    # benchmarking_script (c): effect of warm-up
    uv run python -m systems.sweep --sizes small medium --context-lengths 512 --warmups 0 1 2 5

    # benchmarking_mixed_precision (c): FP32 vs BF16
    uv run python -m systems.sweep --sizes small medium --precisions fp32 bf16

    # torch_compile (b): vanilla vs compiled, full training step
    uv run python -m systems.sweep --sizes small medium --modes forward train --compiled 0 1

    # memory_profiling (b, c): peak memory (see the peak_memory_mib column)
    uv run python -m systems.sweep --sizes small --context-lengths 128 256 512 --modes forward train \
        --precisions fp32 bf16
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

import torch

from systems.benchmark import MODEL_SIZES, MODES, PRECISIONS, run_benchmark
from systems.tables import rows_to_latex, rows_to_markdown


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["small", "medium"], choices=MODEL_SIZES.keys())
    p.add_argument("--context-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    p.add_argument("--modes", nargs="+", default=["forward_backward"], choices=MODES)
    p.add_argument("--precisions", nargs="+", default=["fp32"], choices=PRECISIONS.keys())
    p.add_argument("--warmups", nargs="+", type=int, default=[5])
    p.add_argument("--compiled", nargs="+", type=int, default=[0], choices=[0, 1])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--csv", default=None, help="write all rows to this CSV file")
    p.add_argument("--jsonl", default=None, help="write all rows to this JSONL file")
    p.add_argument("--latex", action="store_true", help="also print LaTeX tables")
    args = p.parse_args()

    rows = []
    for size, ctx, mode, prec, warm, comp in itertools.product(
        args.sizes, args.context_lengths, args.modes, args.precisions, args.warmups, args.compiled
    ):
        print(f"--> {size} ctx={ctx} mode={mode} precision={prec} warmup={warm} compiled={bool(comp)}", flush=True)
        try:
            row = run_benchmark(
                size=size,
                context_length=ctx,
                mode=mode,
                precision=prec,
                warmup_steps=warm,
                steps=args.steps,
                compile_model=bool(comp),
            )
        except torch.cuda.OutOfMemoryError:
            row = dict(
                size=size, context_length=ctx, mode=mode, precision=prec, compiled=bool(comp),
                warmup_steps=warm, error="OOM",
            )
            torch.cuda.empty_cache()
        rows.append(row)

    columns = [
        "size", "context_length", "mode", "precision", "compiled", "warmup_steps",
        "forward_ms_mean", "forward_ms_std", "backward_ms_mean", "backward_ms_std",
        "optimizer_ms_mean", "optimizer_ms_std", "peak_memory_mib", "error",
    ]
    columns = [c for c in columns if any(c in r and r[c] is not None for r in rows)]
    print()
    print(rows_to_markdown(rows, columns))
    if args.latex:
        print()
        print(rows_to_latex(rows, columns))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

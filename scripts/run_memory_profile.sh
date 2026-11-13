#!/usr/bin/env bash
# memory_profiling: snapshot for forward-only and full training step, FP32 and BF16.
# Drop the *.pickle files onto https://pytorch.org/memory_viz and look at "Active Memory Timeline".
set -uo pipefail
mkdir -p memory_snapshots
for ctx in 128 256 512; do
  for mode in forward train; do
    for prec in fp32 bf16; do
      uv run python -m systems.benchmark --size small --context-length "${ctx}" --mode "${mode}" \
        --precision "${prec}" --warmup-steps 3 --steps 1 \
        --memory-snapshot "memory_snapshots/small_${ctx}_${mode}_${prec}.pickle" \
        --output memory_snapshots/results.jsonl
    done
  done
done

#!/usr/bin/env bash
# nsys_profile: profile forward / backward / optimizer step for each model size and context length.
# Warm-up steps are tagged with NVTX ranges "warmup_*"; filter on "measure_*" in the Nsight Systems GUI.
set -uo pipefail
mkdir -p nsys_reports
for size in small medium; do
  for ctx in 128 256 512 1024; do
    echo "== ${size} ctx=${ctx} =="
    uv run nsys profile --force-overwrite true -o "nsys_reports/${size}_${ctx}" \
      --python-backtrace=cuda --pytorch \
      python -m systems.benchmark --size "${size}" --context-length "${ctx}" \
        --mode train --nvtx --warmup-steps 5 --steps 10 \
      || echo "(failed / OOM: ${size} ${ctx})"
  done
done

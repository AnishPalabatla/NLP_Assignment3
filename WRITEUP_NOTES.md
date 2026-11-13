# Writeup notes (analytic answers + where to get the measured ones)

Everything below that is a *measurement* must come from running the scripts on your GPU.
Analytic answers are filled in; measured cells are marked `TODO`.

## How to produce each deliverable

| Problem | Command |
|---|---|
| benchmarking_script (b) | `uv run python -m systems.sweep --sizes small medium --context-lengths 128 256 512 1024` |
| benchmarking_script (c) | `uv run python -m systems.sweep --sizes small medium --context-lengths 512 --warmups 0 1 2 5` |
| nsys_profile | `bash scripts/run_nsys.sh` (filter on NVTX `measure_*`; `forward` / `backward` / `optimizer step` ranges) |
| benchmarking_mixed_precision (c) | `uv run python -m systems.sweep --precisions fp32 bf16 --modes forward_backward` |
| memory_profiling | `bash scripts/run_memory_profile.sh`, then load pickles in https://pytorch.org/memory_viz |
| pytorch_attention | `uv run python -m systems.attention_benchmark` |
| torch_compile (a) | `uv run python -m systems.attention_benchmark --compile` |
| torch_compile (b) | `uv run python -m systems.sweep --modes forward train --compiled 0 1` |
| mixed_precision_accumulation / ToyModel | `uv run python -m systems.mixed_precision` |
| flash_forward / flash_backward | `uv run pytest -k "flash"` |

## benchmarking_script (c) — talking points
Without warm-up the first iterations include one-time costs (CUDA context/cuBLAS handle init, kernel
loading/JIT, caching-allocator growth, autotuning), so the mean is inflated and the std is large.
With only 1–2 warm-up steps some of those costs can still land in the measured window
(e.g. the first backward/optimizer step allocates grads and AdamW state lazily; allocator and
cuBLAS heuristics can still be settling), so numbers remain a bit higher/noisier than with 5.

## mixed_precision_accumulation
Expected outputs: `10.0001` (fp32), `9.9531` (fp16), `10.0021` (fp32 acc + fp16 addend),
`10.0021` (explicit upcast — identical to the previous one).
fp16 accumulation loses precision badly: once the running sum is large, adding 0.01 is below fp16's
resolution and gets rounded away. Keeping the accumulator in fp32 fixes that; the remaining 0.0021 error
is just fp16's representation error of 0.01 (≈0.0100021), not accumulation error.

## benchmarking_mixed_precision (a) — ToyModel under fp16 autocast
- parameters (inside autocast): `float32` (autocast casts per-op copies; stored params are unchanged)
- fc1 output: `float16`
- layer norm output: `float32` (autocast runs layer_norm in fp32)
- logits: `float16` (fc2 is a matmul)
- loss: `float32` (cross-entropy-type losses autocast to fp32)
- gradients: `float32` (they match the fp32 parameters)

(b) LayerNorm's mean/variance reductions (sums of squares, division by a small variance / rsqrt) are the
sensitive parts: they need both range and precision. BF16 has the same exponent range as fp32, so
overflow/underflow is not the issue, but its 7-bit mantissa still makes reductions imprecise, so autocast
still keeps layer norm in fp32 for BF16 — it is just less critical than for FP16.

(c) TODO: fill from the sweep. Typical trend: BF16 gives larger speedups for the larger model (matmuls
dominate and run on tensor cores), while small models see less benefit because of cast overhead and
non-matmul kernels.

## memory_profiling
(d) Residual-stream activation: `batch × context × d_model × 4 B = 4 × L × 768 × 4`.
For L = 512: 6,291,456 B = **6.0 MiB** (L=128: 1.5 MiB, L=256: 3.0 MiB).

(e) Expect the largest forward allocation to be the logits `4 × L × 10000 × 4 B` (L=512: 78.1 MiB),
coming from the `lm_head` linear layer (and its softmax/cross-entropy temporaries in a training step);
for L=512 the next-largest are the attention score matrices `4 × 12 × 512 × 512 × 4 B = 48 MiB`.
Confirm against your snapshot / stack traces.

(a), (b), (c): TODO — from the snapshots / `peak_memory_mib` column.

## pytorch_attention — memory accounting
Batch 8, no heads, sequence length n, d_model d, fp32. One n×n score tensor is `8 · n² · 4 B`:
n = 8192 → 2 GiB, n = 16384 → 8 GiB (independent of d). The basics implementation materialises several of them
(scores, scaled scores, softmax intermediates), and autograd keeps at least the exp output and the attention
weights for backward, i.e. ≥ 2·n² floats per batch element, while Q, K, V and the output are only `O(n·d)`.
So the memory saved for backward grows **quadratically** in sequence length (×4 per doubling) and dominates
once n ≫ d. TODO: report your GPU's OOM point and plug in the numbers for the smallest OOM config.
Fix: never materialise P — tile the computation (FlashAttention) and recompute S/P in the backward pass from
Q, K and the saved logsumexp L (O(n) extra memory instead of O(n²)).

## nsys_profile (e) — talking point
Per head, the two attention matmuls cost ≈ 2·n²·d_k FLOPs each, while softmax is only a few FLOPs per element
(O(n²)), so softmax has far fewer FLOPs than the matmuls — yet it takes a disproportionately large share of
runtime because it is memory-bandwidth bound (several passes over the n×n matrix: max, subtract, exp, sum, divide).

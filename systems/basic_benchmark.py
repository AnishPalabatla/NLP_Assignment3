"""Benchmark forward-only or forward-and-backward execution of BasicsTransformerLM."""

from __future__ import annotations

import argparse
import json
import statistics
import timeit

import torch
import torch.nn.functional as F

from basics.model import BasicsTransformerLM


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=positive_int, default=10_000)
    parser.add_argument("--context-length", type=positive_int, default=512)
    parser.add_argument("--d-model", type=positive_int, default=768)
    parser.add_argument("--num-layers", type=positive_int, default=12)
    parser.add_argument("--num-heads", type=positive_int, default=12)
    parser.add_argument("--d-ff", type=positive_int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=5)
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument("--mode", choices=("forward", "forward_backward"), default="forward_backward")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("the head dimension (d-model / num-heads) must be even for rotary embeddings")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    tokens = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    forward_times: list[float] = []
    backward_times: list[float] = []

    def run_step(measure: bool) -> None:
        if args.mode == "forward_backward":
            model.zero_grad(set_to_none=True)
            forward_start = timeit.default_timer()
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
            synchronize(device)
            forward_end = timeit.default_timer()

            backward_start = timeit.default_timer()
            loss.backward()
            synchronize(device)
            backward_end = timeit.default_timer()

            if measure:
                forward_times.append(forward_end - forward_start)
                backward_times.append(backward_end - backward_start)
        else:
            forward_start = timeit.default_timer()
            with torch.no_grad():
                model(tokens)
            synchronize(device)
            forward_end = timeit.default_timer()
            if measure:
                forward_times.append(forward_end - forward_start)

    for _ in range(args.warmup_steps):
        run_step(measure=False)
    synchronize(device)

    for _ in range(args.steps):
        run_step(measure=True)

    def summarize(samples: list[float]) -> dict[str, float]:
        values_ms = [sample * 1_000 for sample in samples]
        return {
            "mean_ms": statistics.fmean(values_ms),
            "std_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
        }

    result = {
        "mode": args.mode,
        "device": str(device),
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "forward": summarize(forward_times),
    }
    if backward_times:
        result["backward"] = summarize(backward_times)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

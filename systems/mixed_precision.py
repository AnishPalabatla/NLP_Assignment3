"""Small experiments for `mixed_precision_accumulation` and `benchmarking_mixed_precision (a)`.

    uv run python -m systems.mixed_precision
"""

from __future__ import annotations

import torch
import torch.nn as nn


def accumulation_demo() -> None:
    print("== mixed_precision_accumulation ==")
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("fp32 accumulator, fp32 addend :", s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp16 accumulator, fp16 addend :", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp32 accumulator, fp16 addend :", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("fp32 accumulator, fp16 -> fp32:", s)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def toy_model_dtypes(dtype: torch.dtype = torch.float16) -> None:
    print(f"\n== ToyModel dtypes under autocast({dtype}) ==")
    device = "cuda"
    model = ToyModel(20, 5).to(device)
    x = torch.randn(8, 20, device=device)
    y = torch.randint(0, 5, (8,), device=device)

    captured = {}
    model.fc1.register_forward_hook(lambda m, i, o: captured.__setitem__("fc1 output", o.dtype))
    model.ln.register_forward_hook(lambda m, i, o: captured.__setitem__("layer norm output", o.dtype))

    with torch.autocast(device_type="cuda", dtype=dtype):
        captured["parameters (inside autocast)"] = next(model.parameters()).dtype
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
    loss.backward()

    captured["logits"] = logits.dtype
    captured["loss"] = loss.dtype
    captured["gradients"] = model.fc1.weight.grad.dtype
    for k, v in captured.items():
        print(f"{k:32s}: {v}")


if __name__ == "__main__":
    accumulation_demo()
    if torch.cuda.is_available():
        toy_model_dtypes(torch.float16)
        toy_model_dtypes(torch.bfloat16)
    else:
        print("\n(CUDA not available: skipping ToyModel autocast dtype check)")

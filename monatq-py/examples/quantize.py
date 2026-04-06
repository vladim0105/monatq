"""
PyTorch PT2E quantization using a monatq-backed observer.

MonatqObserver replaces the standard MinMaxObserver in a PT2E calibration
workflow. Instead of tracking a fixed-size histogram, it uses a T-Digest
(via monatq.TensorDigest) to estimate the activation range at an arbitrary
clip_percentile — more memory-efficient and accurate in the tails.

Usage:
    python examples/quantize.py

Requirements:
    pip install -e ".[dev]"   # installs torch and monatq
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from torch.ao.quantization.observer import ObserverBase, _ObserverBase
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from monatq import TensorDigest


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """Small 3-channel CNN for 32×32 inputs, 10 output classes."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1))


def train_model(model: nn.Module, steps: int = 20, batch: int = 64) -> None:
    """Quick synthetic training to get non-trivial BN stats and weights."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(steps):
        x = torch.randn(batch, 3, 32, 32)
        y = torch.randint(0, 10, (batch,))
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# MonatqObserver
# ---------------------------------------------------------------------------

class MonatqObserver(_ObserverBase):
    """
    Activation observer backed by monatq TensorDigest.

    On each forward pass the activation tensor is flattened and fed into a
    TensorDigest. After calibration, calculate_qparams() merges all per-position
    digests into one global distribution, then queries at clip_percentile and
    (1 - clip_percentile) for a range estimate that clips outliers.

    Args:
        clip_percentile: fraction to clip from each tail.  0.0 = exact min/max.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.qint8,
        qscheme: torch.qscheme = torch.per_tensor_affine,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        clip_percentile: float = 0.001,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            **kwargs,
        )
        self._digest: Optional[TensorDigest] = None
        self._clip = clip_percentile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.detach().cpu().float().contiguous().view(-1)
        if self._digest is None:
            self._digest = TensorDigest([x_cpu.numel()], compression=100)
        self._digest.update(x_cpu)
        return x

    def calculate_qparams(self):
        if self._digest is None:
            return torch.tensor([1.0]), torch.tensor([0], dtype=torch.int32)
        merged = self._digest.merge_all()
        min_val = torch.tensor([merged.quantile(self._clip)[0]])
        max_val = torch.tensor([merged.quantile(1.0 - self._clip)[0]])
        return self._calculate_qparams(min_val, max_val)


# ---------------------------------------------------------------------------
# Quantization workflow
# ---------------------------------------------------------------------------

def replace_observers(model: nn.Module) -> int:
    """
    Replace all per-tensor ObserverBase instances in a prepared PT2E model
    with MonatqObserver.  Per-channel weight observers are left unchanged.
    Returns the number of observers replaced.
    """
    replaced = 0
    for name, mod in list(model.named_modules()):
        if (
            isinstance(mod, ObserverBase)
            and not isinstance(mod, MonatqObserver)
            and getattr(mod, "qscheme", None) == torch.per_tensor_affine
        ):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent,
                child_name,
                MonatqObserver(
                    dtype=mod.dtype,
                    qscheme=mod.qscheme,
                    quant_min=mod.quant_min,
                    quant_max=mod.quant_max,
                    clip_percentile=0.001,
                ),
            )
            replaced += 1
    return replaced


def build_quantized_model(
    model: nn.Module,
    calib_batches: int = 50,
    batch_size: int = 32,
) -> nn.Module:
    model.eval()
    # Export with the same batch size used for calibration/evaluation to avoid
    # dynamic-shape constraints from AdaptiveAvgPool2d.
    example = torch.randn(batch_size, 3, 32, 32)

    print("Exporting model...")
    ep = torch.export.export(model, (example,))

    print("Preparing PT2E model with XNNPACKQuantizer...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.filterwarnings("ignore", message="erase_node")
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        prepared = prepare_pt2e(ep.module(), quantizer)
        n = replace_observers(prepared)
        print(f"Replaced {n} activation observers with MonatqObserver (clip_percentile=0.001)")

        total_samples = calib_batches * batch_size
        print(f"Calibrating ({calib_batches} batches × {batch_size} = {total_samples} samples)...")
        torch.ao.quantization.move_exported_model_to_eval(prepared)
        with torch.no_grad():
            for _ in range(calib_batches):
                prepared(torch.randn(batch_size, 3, 32, 32))

        print("Converting to quantized model...")
        return convert_pt2e(prepared)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    float_model: nn.Module,
    quant_model: nn.Module,
    n_batches: int = 200,
    batch_size: int = 32,
) -> None:
    float_model.eval()
    torch.ao.quantization.move_exported_model_to_eval(quant_model)

    total = 0
    correct_f = correct_q = 0
    signal_power = noise_power = mse_sum = 0.0

    for _ in range(n_batches):
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))

        logits_f = float_model(x)
        logits_q = quant_model(x)

        correct_f += (logits_f.argmax(1) == y).sum().item()
        correct_q += (logits_q.argmax(1) == y).sum().item()
        total += batch_size

        diff = logits_f - logits_q
        mse_sum += diff.pow(2).mean().item()
        signal_power += logits_f.pow(2).mean().item()
        noise_power += diff.pow(2).mean().item()

    mse = mse_sum / n_batches
    sqnr = 10.0 * math.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    acc_f = correct_f / total
    acc_q = correct_q / total

    print()
    print("=" * 60)
    print(" monatq PT2E Quantization Example")
    print("=" * 60)
    print(f" Model:       TinyCNN (3→16→32→10 classes)")
    print(f" Calibration: 50 batches × 32 = 1600 samples")
    print(f" Observer:    MonatqObserver (clip_percentile=0.001)")
    print("=" * 60)
    print(f" Float  top-1:   {acc_f:.4f}")
    print(f" Quant  top-1:   {acc_q:.4f}")
    print(f" Δ accuracy:     {acc_q - acc_f:+.4f}")
    print(f" MSE (logits):   {mse:.6f}")
    print(f" SQNR:           {sqnr:.2f} dB")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    print("Training TinyCNN on synthetic data (20 steps)...")
    model = TinyCNN()
    train_model(model, steps=20)

    quantized = build_quantized_model(model)
    evaluate(model, quantized)


if __name__ == "__main__":
    main()

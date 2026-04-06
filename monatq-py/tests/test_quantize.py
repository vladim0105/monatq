"""
End-to-end smoke test for examples/quantize.py.
"""

import math
import sys
import os
import warnings

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../examples"))
import quantize as Q  # noqa: E402


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    model = Q.TinyCNN()
    Q.train_model(model, steps=3, batch=4)
    quantized = Q.build_quantized_model(model, calib_batches=3, batch_size=4)
    return model, quantized


def test_quantization_smoke(models):
    float_model, quant_model = models
    float_model.eval()
    torch.ao.quantization.move_exported_model_to_eval(quant_model)

    signal = noise = 0.0
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(4, 3, 32, 32)
            lf = float_model(x)
            lq = quant_model(x)
            assert lf.shape == lq.shape == (4, 10)
            diff = lf - lq
            signal += lf.pow(2).mean().item()
            noise += diff.pow(2).mean().item()

    sqnr = 10.0 * math.log10(signal / noise) if noise > 0 else float("inf")
    assert sqnr >= 20.0, f"SQNR {sqnr:.1f} dB is below 20 dB"

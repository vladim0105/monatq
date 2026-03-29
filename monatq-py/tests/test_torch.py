"""
Pytest tests for monatq.TensorDigest with PyTorch tensors.
Covers both float32 and int32 dtypes.
"""

import pytest
import tempfile
import os
import torch
from monatq import TensorDigest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ramp_f32(n: int) -> torch.Tensor:
    """1-D float32 tensor with values [0, 1/n, 2/n, ..., (n-1)/n]."""
    return torch.linspace(0.0, 1.0, n, dtype=torch.float32)


def make_ramp_i32(n: int) -> torch.Tensor:
    """1-D int32 tensor with values [0, 1, 2, ..., n-1]."""
    return torch.arange(n, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_dtype_is_float32(self):
        td = TensorDigest([4], 100)
        assert td.dtype == "float32"

    def test_explicit_float32_str(self):
        td = TensorDigest([4], 100, dtype="float32")
        assert td.dtype == "float32"

    def test_explicit_int32_str(self):
        td = TensorDigest([4], 100, dtype="int32")
        assert td.dtype == "int32"

    def test_torch_float32(self):
        td = TensorDigest([4], 100, dtype=torch.float32)
        assert td.dtype == "float32"

    def test_torch_int32(self):
        td = TensorDigest([4], 100, dtype=torch.int32)
        assert td.dtype == "int32"

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="cannot interpret"):
            TensorDigest([4], 100, dtype="float64")

    def test_shape_preserved(self):
        td = TensorDigest([2, 3, 4], 50)
        assert td.shape == [2, 3, 4]
        assert td.numel == 24


# ---------------------------------------------------------------------------
# float32 tensors
# ---------------------------------------------------------------------------

class TestFloat32:
    def test_basic_update_and_quantile(self):
        td = TensorDigest([3], 100)
        for i in range(1000):
            x = float(i)
            td.update(torch.tensor([x * 0.006283, (x * 0.006283 + 1.5707), x / 1000.0],
                                   dtype=torch.float32))
        q50 = td.quantile(0.5)
        assert len(q50) == 3
        # ramp 0..1 median ≈ 0.5
        assert abs(q50[2] - 0.5) < 0.05

    def test_median_of_ramp(self):
        td = TensorDigest([1], 100)
        for i in range(1000):
            td.update(torch.tensor([i / 999.0], dtype=torch.float32))
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 0.5) < 0.05

    def test_min_max_quantiles(self):
        td = TensorDigest([1], 100)
        for v in [0.0, 1.0, 2.0, 3.0]:
            td.update(torch.tensor([v], dtype=torch.float32))
        assert td.quantile(0.0)[0] == pytest.approx(0.0, abs=0.01)
        assert td.quantile(1.0)[0] == pytest.approx(3.0, abs=0.01)

    def test_multidim_shape(self):
        shape = [2, 3]
        td = TensorDigest(shape, 50)
        t = torch.zeros(2, 3, dtype=torch.float32)
        td.update(t)
        assert td.numel == 6

    def test_non_contiguous_raises(self):
        td = TensorDigest([3], 100)
        t = torch.arange(6, dtype=torch.float32)[::2]  # stride > 1 → non-contiguous
        assert not t.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            td.update(t)

    def test_wrong_dtype_raises(self):
        td = TensorDigest([3], 100, dtype="float32")
        with pytest.raises(ValueError, match="float32"):
            td.update(torch.zeros(3, dtype=torch.int32))

    def test_wrong_numel_raises(self):
        td = TensorDigest([3], 100)
        with pytest.raises((ValueError, RuntimeError)):
            td.update(torch.zeros(5, dtype=torch.float32))

    def test_multiple_quantiles(self):
        td = TensorDigest([1], 100)
        for i in range(1000):
            td.update(torch.tensor([float(i)], dtype=torch.float32))
        qs = td.quantiles([0.25, 0.5, 0.75])
        assert len(qs) == 3
        assert qs[0][0] < qs[1][0] < qs[2][0]

    def test_flush_then_quantile(self):
        td = TensorDigest([1], 100)
        for i in range(10):
            td.update(torch.tensor([float(i)], dtype=torch.float32))
        td.flush()
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 4.5) < 1.0


# ---------------------------------------------------------------------------
# int32 tensors
# ---------------------------------------------------------------------------

class TestInt32:
    def test_basic_update_and_quantile(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(1000):
            td.update(torch.tensor([i], dtype=torch.int32))
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 499.5) < 5.0

    def test_median_matches_expected(self):
        td = TensorDigest([1], 100, dtype="int32")
        values = torch.arange(0, 100, dtype=torch.int32)
        for v in values:
            td.update(v.unsqueeze(0))
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 49.5) < 2.0

    def test_min_max_quantiles(self):
        td = TensorDigest([1], 100, dtype="int32")
        for v in [0, 1, 2, 100]:
            td.update(torch.tensor([v], dtype=torch.int32))
        assert td.quantile(0.0)[0] == pytest.approx(0.0, abs=0.5)
        assert td.quantile(1.0)[0] == pytest.approx(100.0, abs=0.5)

    def test_multidim_shape(self):
        shape = [2, 3]
        td = TensorDigest(shape, 50, dtype="int32")
        t = torch.zeros(2, 3, dtype=torch.int32)
        td.update(t)
        assert td.numel == 6

    def test_wrong_dtype_raises(self):
        td = TensorDigest([3], 100, dtype="int32")
        with pytest.raises(ValueError, match="int32"):
            td.update(torch.zeros(3, dtype=torch.float32))

    def test_non_contiguous_raises(self):
        td = TensorDigest([3], 100, dtype="int32")
        t = torch.arange(6, dtype=torch.int32)[::2]
        assert not t.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            td.update(t)

    def test_wrong_numel_raises(self):
        td = TensorDigest([3], 100, dtype="int32")
        with pytest.raises((ValueError, RuntimeError)):
            td.update(torch.zeros(5, dtype=torch.int32))

    def test_multiple_quantiles(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(1000):
            td.update(torch.tensor([i], dtype=torch.int32))
        qs = td.quantiles([0.25, 0.5, 0.75])
        assert len(qs) == 3
        assert qs[0][0] < qs[1][0] < qs[2][0]

    def test_flush_then_quantile(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(10):
            td.update(torch.tensor([i], dtype=torch.int32))
        td.flush()
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 4.5) < 1.0

    def test_quantile_dtype_is_float32(self):
        """Quantile results are always f32 regardless of input dtype."""
        td = TensorDigest([1], 100, dtype="int32")
        td.update(torch.tensor([42], dtype=torch.int32))
        result = td.quantile(0.5)
        assert isinstance(result[0], float)


# ---------------------------------------------------------------------------
# Save / load (dtype auto-detection)
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_f32_roundtrip_no_dtype(self):
        td = TensorDigest([1], 100, dtype="float32")
        for i in range(200):
            td.update(torch.tensor([float(i)], dtype=torch.float32))
        q50_before = td.quantile(0.5)[0]

        with tempfile.NamedTemporaryFile(suffix=".monatq", delete=False) as f:
            path = f.name
        try:
            td.save(path)
            loaded = TensorDigest.load(path)
            assert loaded.dtype == "float32"
            assert abs(loaded.quantile(0.5)[0] - q50_before) < 0.5
        finally:
            os.unlink(path)

    def test_i32_roundtrip_no_dtype(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(200):
            td.update(torch.tensor([i], dtype=torch.int32))
        q50_before = td.quantile(0.5)[0]

        with tempfile.NamedTemporaryFile(suffix=".monatq", delete=False) as f:
            path = f.name
        try:
            td.save(path)
            loaded = TensorDigest.load(path)
            assert loaded.dtype == "int32"
            assert abs(loaded.quantile(0.5)[0] - q50_before) < 0.5
        finally:
            os.unlink(path)

    def test_shape_preserved_after_load(self):
        td = TensorDigest([2, 3], 50)
        td.update(torch.zeros(2, 3, dtype=torch.float32))

        with tempfile.NamedTemporaryFile(suffix=".monatq", delete=False) as f:
            path = f.name
        try:
            td.save(path)
            loaded = TensorDigest.load(path)
            assert loaded.shape == [2, 3]
            assert loaded.numel == 6
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Cross-dtype accuracy comparison
# ---------------------------------------------------------------------------

class TestDtypeConsistency:
    """float32 and int32 digests fed the same integer values should give
    similar quantile estimates."""

    def test_median_agrees(self):
        n = 500
        td_f32 = TensorDigest([1], 100, dtype="float32")
        td_i32 = TensorDigest([1], 100, dtype="int32")

        for i in range(n):
            td_f32.update(torch.tensor([float(i)], dtype=torch.float32))
            td_i32.update(torch.tensor([i], dtype=torch.int32))

        q50_f32 = td_f32.quantile(0.5)[0]
        q50_i32 = td_i32.quantile(0.5)[0]
        assert abs(q50_f32 - q50_i32) < 2.0

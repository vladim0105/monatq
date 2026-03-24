"""
Pytest tests for monatq.TensorDigest with NumPy arrays.
Covers both float32 and int32 dtypes (both go through the buffer protocol path).

Note: wrong-dtype and non-contiguous tests are absent — numpy silently handles
both via the buffer protocol, so there is no error to assert.
"""

import pytest
import tempfile
import os
import numpy as np
from monatq import TensorDigest


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

    def test_numpy_float32_type(self):
        td = TensorDigest([4], 100, dtype=np.float32)
        assert td.dtype == "float32"

    def test_numpy_int32_type(self):
        td = TensorDigest([4], 100, dtype=np.int32)
        assert td.dtype == "int32"

    def test_numpy_dtype_instance_float32(self):
        td = TensorDigest([4], 100, dtype=np.dtype("float32"))
        assert td.dtype == "float32"

    def test_numpy_dtype_instance_int32(self):
        td = TensorDigest([4], 100, dtype=np.dtype("int32"))
        assert td.dtype == "int32"

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="cannot interpret"):
            TensorDigest([4], 100, dtype="float64")

    def test_shape_preserved(self):
        td = TensorDigest([2, 3, 4], 50)
        assert td.shape == [2, 3, 4]
        assert td.numel == 24


# ---------------------------------------------------------------------------
# float32 arrays
# ---------------------------------------------------------------------------

class TestFloat32:
    def test_basic_update_and_quantile(self):
        td = TensorDigest([3], 100)
        for i in range(1000):
            x = float(i)
            td.update(np.array([x * 0.006283, x * 0.006283 + 1.5707, x / 1000.0],
                               dtype=np.float32))
        q50 = td.quantile(0.5)
        assert len(q50) == 3
        # ramp 0..1 median ≈ 0.5
        assert abs(q50[2] - 0.5) < 0.05

    def test_median_of_ramp(self):
        td = TensorDigest([1], 100)
        for i in range(1000):
            td.update(np.array([i / 999.0], dtype=np.float32))
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 0.5) < 0.05

    def test_min_max_quantiles(self):
        td = TensorDigest([1], 100)
        for v in [0.0, 1.0, 2.0, 3.0]:
            td.update(np.array([v], dtype=np.float32))
        assert td.quantile(0.0)[0] == pytest.approx(0.0, abs=0.01)
        assert td.quantile(1.0)[0] == pytest.approx(3.0, abs=0.01)

    def test_multidim_shape(self):
        td = TensorDigest([2, 3], 50)
        td.update(np.zeros((2, 3), dtype=np.float32))
        assert td.numel == 6

    def test_wrong_numel_raises(self):
        td = TensorDigest([3], 100)
        with pytest.raises((ValueError, RuntimeError)):
            td.update(np.zeros(5, dtype=np.float32))

    def test_multiple_quantiles(self):
        td = TensorDigest([1], 100)
        for i in range(1000):
            td.update(np.array([float(i)], dtype=np.float32))
        qs = td.quantiles([0.25, 0.5, 0.75])
        assert len(qs) == 3
        assert qs[0][0] < qs[1][0] < qs[2][0]

    def test_flush_then_quantile(self):
        td = TensorDigest([1], 100)
        for i in range(10):
            td.update(np.array([float(i)], dtype=np.float32))
        td.flush()
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 4.5) < 1.0


# ---------------------------------------------------------------------------
# int32 arrays
# ---------------------------------------------------------------------------

class TestInt32:
    def test_basic_update_and_quantile(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(1000):
            td.update(np.array([i], dtype=np.int32))
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 499.5) < 5.0

    def test_median_matches_expected(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(100):
            td.update(np.array([i], dtype=np.int32))
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 49.5) < 2.0

    def test_min_max_quantiles(self):
        td = TensorDigest([1], 100, dtype="int32")
        for v in [0, 1, 2, 100]:
            td.update(np.array([v], dtype=np.int32))
        assert td.quantile(0.0)[0] == pytest.approx(0.0, abs=0.5)
        assert td.quantile(1.0)[0] == pytest.approx(100.0, abs=0.5)

    def test_multidim_shape(self):
        td = TensorDigest([2, 3], 50, dtype="int32")
        td.update(np.zeros((2, 3), dtype=np.int32))
        assert td.numel == 6

    def test_wrong_numel_raises(self):
        td = TensorDigest([3], 100, dtype="int32")
        with pytest.raises((ValueError, RuntimeError)):
            td.update(np.zeros(5, dtype=np.int32))

    def test_multiple_quantiles(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(1000):
            td.update(np.array([i], dtype=np.int32))
        qs = td.quantiles([0.25, 0.5, 0.75])
        assert len(qs) == 3
        assert qs[0][0] < qs[1][0] < qs[2][0]

    def test_flush_then_quantile(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(10):
            td.update(np.array([i], dtype=np.int32))
        td.flush()
        q50 = td.quantile(0.5)
        assert abs(q50[0] - 4.5) < 1.0

    def test_quantile_dtype_is_float32(self):
        """Quantile results are always f32 regardless of input dtype."""
        td = TensorDigest([1], 100, dtype="int32")
        td.update(np.array([42], dtype=np.int32))
        result = td.quantile(0.5)
        assert isinstance(result[0], float)


# ---------------------------------------------------------------------------
# Save / load (dtype auto-detection)
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_f32_roundtrip(self):
        td = TensorDigest([1], 100, dtype="float32")
        for i in range(200):
            td.update(np.array([float(i)], dtype=np.float32))
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

    def test_i32_roundtrip(self):
        td = TensorDigest([1], 100, dtype="int32")
        for i in range(200):
            td.update(np.array([i], dtype=np.int32))
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
        td.update(np.zeros((2, 3), dtype=np.float32))

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
            td_f32.update(np.array([float(i)], dtype=np.float32))
            td_i32.update(np.array([i], dtype=np.int32))

        q50_f32 = td_f32.quantile(0.5)[0]
        q50_i32 = td_i32.quantile(0.5)[0]
        assert abs(q50_f32 - q50_i32) < 2.0

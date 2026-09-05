"""
Kernel selection, the RankKnot default, and the operations it newly supports.
"""

import os
import tempfile

import numpy as np
import pytest
from monatq import TensorDigest


class TestKernelSelection:
    def test_default_kernel_is_rankknot(self):
        td = TensorDigest([4])
        assert td.kernel == "rankknot"

    def test_kernel_can_be_named_explicitly(self):
        assert TensorDigest([4], kernel="tdigest").kernel == "tdigest"
        assert TensorDigest([4], kernel="rankknot").kernel == "rankknot"

    def test_kernel_name_is_case_and_space_insensitive(self):
        assert TensorDigest([4], kernel="  RankKnot ").kernel == "rankknot"

    def test_unknown_kernel_is_rejected(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            TensorDigest([4], kernel="unknown")

    def test_both_dtypes_work_on_both_kernels(self):
        for kernel in ("rankknot", "tdigest"):
            for dtype in ("float32", "int32"):
                td = TensorDigest([2], kernel=kernel, dtype=dtype)
                assert td.kernel == kernel
                assert td.dtype == dtype


class TestConfigKnobs:
    def test_compression_is_rejected_for_rankknot(self):
        # Silently ignoring an accuracy knob is the failure mode this prevents.
        with pytest.raises(ValueError, match="compression is a tdigest setting"):
            TensorDigest([4], kernel="rankknot", compression=100)

    def test_buffer_capacity_is_rejected_for_tdigest(self):
        with pytest.raises(ValueError, match="buffer_capacity is a rankknot setting"):
            TensorDigest([4], kernel="tdigest", buffer_capacity=64)

    def test_buffer_capacity_is_accepted_for_rankknot(self):
        td = TensorDigest([1], kernel="rankknot", buffer_capacity=8)
        for value in range(100):
            td.update(np.array([float(value)], dtype=np.float32))
        assert td.quantile(0.0)[0] == pytest.approx(0.0)
        assert td.quantile(1.0)[0] == pytest.approx(99.0)

    def test_zero_buffer_capacity_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            TensorDigest([4], kernel="rankknot", buffer_capacity=0)

    def test_shape_is_still_positional(self):
        assert TensorDigest([2, 3]).shape == [2, 3]


def _filled(kernel, shape=(2, 2), n=400):
    td = TensorDigest(list(shape), kernel=kernel)
    rng = np.random.default_rng(0)
    for _ in range(n):
        td.update(rng.normal(size=shape).astype(np.float32))
    return td


class TestRankKnotOperations:
    """Operations that reported NotImplementedError for RankKnot until now."""

    def test_analyze_works_on_the_default_kernel(self):
        td = TensorDigest([1], kernel="rankknot")
        rng = np.random.default_rng(1)
        for _ in range(4000):
            td.update(np.array([rng.normal()], dtype=np.float32))
        assert td.analyze() == ["Normal"]

    def test_merge_all_works_on_the_default_kernel(self):
        merged = _filled("rankknot").merge_all()
        assert merged.numel == 1
        assert merged.kernel == "rankknot"

    def test_without_zeros_recovers_spread_behind_a_zero_spike(self):
        td = TensorDigest([1], kernel="rankknot")
        rng = np.random.default_rng(2)
        for i in range(4000):
            value = 0.0 if i % 5 else rng.normal()
            td.update(np.array([value], dtype=np.float32))

        assert td.quantile(0.25)[0] == pytest.approx(0.0)
        assert td.quantile(0.75)[0] == pytest.approx(0.0)

        filtered = td.without_zeros()
        iqr = filtered.quantile(0.75)[0] - filtered.quantile(0.25)[0]
        assert iqr == pytest.approx(1.349, abs=0.25)

    def test_visualize_is_available(self):
        # Only that it is no longer NotImplementedError; the server itself is covered in Rust.
        assert hasattr(TensorDigest([1], kernel="rankknot"), "visualize")


class TestSnapshots:
    @pytest.mark.parametrize("kernel", ["rankknot", "tdigest"])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_load_detects_kernel_and_dtype(self, kernel, dtype):
        td = TensorDigest([3], kernel=kernel, dtype=dtype)
        np_dtype = np.float32 if dtype == "float32" else np.int32
        for value in range(200):
            td.update(np.array([value, -value, 1], dtype=np_dtype))
        expected = td.quantile(0.5)

        path = os.path.join(tempfile.gettempdir(), f"monatq_{kernel}_{dtype}.bin")
        try:
            td.save(path)
            restored = TensorDigest.load(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

        assert restored.kernel == kernel
        assert restored.dtype == dtype
        assert restored.quantile(0.5) == pytest.approx(expected)

    def test_to_bytes_and_from_bytes_round_trip(self):
        td = _filled("rankknot")
        expected = td.quantile(0.5)
        restored = TensorDigest.from_bytes(td.to_bytes())
        assert restored.kernel == "rankknot"
        assert restored.quantile(0.5) == pytest.approx(expected)

    def test_corrupt_bytes_raise_value_error(self):
        with pytest.raises(ValueError):
            TensorDigest.from_bytes(b"not a snapshot")

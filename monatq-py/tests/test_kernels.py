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

    def test_zero_buffer_capacity_is_accepted(self):
        td = TensorDigest([1], kernel="rankknot", buffer_capacity=0)
        for value in range(100):
            td.update(np.array([float(value)], dtype=np.float32))
        assert td.quantile(0.0)[0] == pytest.approx(0.0)
        assert td.quantile(1.0)[0] == pytest.approx(99.0)

    def test_shape_is_still_positional(self):
        digest = TensorDigest([2, 3])
        assert digest.shape == [2, 3]
        assert digest.block_count == 6
        assert not hasattr(digest, "numel")
        assert not hasattr(digest, "block_shape")

    @pytest.mark.parametrize("kernel", ["rankknot", "tdigest"])
    def test_block_pooling_exposes_atomic_block_geometry(self, kernel):
        td = TensorDigest([2, 5, 2], kernel=kernel, blocks_per_axis=2, block_axis=1)
        assert td.shape == [2, 2, 2]
        assert td.block_count == 8
        assert td.input_shape == [2, 5, 2]
        assert td.input_numel == 20
        assert td.blocks_per_axis == 2
        assert td.block_axis == 1
        for step in range(3):
            row = np.arange(20, dtype=np.float32) + step * 100
            td.update(row.reshape(2, 5, 2))
        assert len(td.quantile(0.5)) == 8
        assert td.cell_quantiles(0, [0.0, 1.0]) != td.cell_quantiles(2, [0.0, 1.0])
        with pytest.raises(ValueError, match="element count"):
            td.update(np.zeros(td.block_count, dtype=np.float32))
        with pytest.raises(IndexError):
            td.cell_quantiles(td.block_count, [0.5])

    @pytest.mark.parametrize("kernel", ["rankknot", "tdigest"])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_default_and_negative_block_axes_and_snapshot(self, kernel, dtype):
        td = TensorDigest([2, 5], kernel=kernel, dtype=dtype, blocks_per_axis=2)
        negative = TensorDigest(
            [2, 5], kernel=kernel, dtype=dtype, blocks_per_axis=2, block_axis=-1
        )
        assert td.shape == negative.shape == [2, 2]
        row = np.arange(10, dtype=dtype).reshape(2, 5)
        td.update(row)
        restored = TensorDigest.from_bytes(td.to_bytes())
        restored.update(row + 10)
        assert restored.blocks_per_axis == 2
        assert restored.block_axis == 1
        assert restored.quantile(1.0) == [12.0, 14.0, 17.0, 19.0]
        assert TensorDigest([2, 5], block_axis=-2).blocks_per_axis == 0

    def test_requested_block_count_modes_and_scalar_default(self):
        assert TensorDigest([]).blocks_per_axis == 0
        default = TensorDigest([2, 5])
        assert default.blocks_per_axis == 0
        assert default.block_axis == 1
        assert TensorDigest([2, 5], blocks_per_axis=0).shape == [2, 5]
        whole = TensorDigest([2, 5], blocks_per_axis=1)
        clamped = TensorDigest([2, 5], blocks_per_axis=99)
        assert whole.shape == [2, 1]
        assert clamped.shape == [2, 5]
        assert clamped.blocks_per_axis == 99
        with pytest.raises(TypeError):
            TensorDigest([2, 5], block_size=2)

    def test_block_arguments_are_validated(self):
        with pytest.raises(ValueError, match="block_axis"):
            TensorDigest([2, 3], blocks_per_axis=2, block_axis=-3)
        with pytest.raises(ValueError, match="block_axis"):
            TensorDigest([2, 3], blocks_per_axis=2, block_axis=2)
        td = TensorDigest([2, 3], blocks_per_axis=0, block_axis=1)
        assert td.shape == [2, 3]
        assert td.blocks_per_axis == 0


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
        assert merged.block_count == 1
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

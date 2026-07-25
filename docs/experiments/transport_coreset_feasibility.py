#!/usr/bin/env python3
"""Feasibility test for a tie-aware rank-transport coreset.

Two 400-byte summary-state layouts are tested. This figure excludes the batch
input buffer and flush workspace. RTC-48 uses 48 packed (f32 value, u32 count)
knots plus a purity mask and metadata. RTC-64-u16 evaluates the interior coreset
used by the proposed layout: 64 f32 values, 64 u16 probability masses, and a
purity mask. The final eight bytes are exact f32 min/max sidecars; the sample
count is shared by the parent tensor and is simulated by the local `seen`
variable. The queried grid excludes q=0 and q=1, so those sidecars do not alter
reported rows. Prefix rounding bounds each current mass-CDF boundary error by
1/(2*65535) per requantization.
There is no separately sized atom table. Equal input values are indivisible;
any number of distinct values competes for the general knot budget.

This prototype is deliberately simple. Every 256-row flush unions exact batch
runs with the old coreset, places boundaries on a fixed arcsine rank scale,
snaps each boundary to the nearest indivisible mass boundary, and stores the
weighted mean of each segment. If several cuts collapse inside one large tie,
the unused knots recursively split the widest remaining rank intervals. A
one-value segment remains "pure". Queries use
linear quantile interpolation between mixed knots and a horizontal interval
for a pure knot, so retained ties are exact.

The offline variant compresses the complete empirical measure once and is an
information/layout upper bound, not a streaming algorithm. Parameters were
selected on seed 40; reported rows use held-out seeds 44 and 55.

Usage:
    python3 docs/experiments/transport_coreset_feasibility.py

Requirements: Python 3.11+, NumPy, Cargo. Rust baseline build files are created
under the system temporary directory.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KNOTS = 48
QUANTIZED_KNOTS = 64
MASS_QUANTA = 65_535
BATCH_ROWS = 256
QS = np.unique(
    np.r_[
        np.geomspace(1e-4, 1e-2, 20),
        np.linspace(1e-2, 1 - 1e-2, 197),
        1 - np.geomspace(1e-4, 1e-2, 20),
    ].astype(np.float32)
).astype(np.float64)
TAIL = (QS <= .01) | (QS >= .99)


@dataclass
class Coreset:
    values: np.ndarray
    weights: np.ndarray
    pure: np.ndarray


def empty() -> Coreset:
    return Coreset(
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
    )


def exact_runs(values: np.ndarray) -> Coreset:
    unique, counts = np.unique(np.asarray(values, dtype=np.float32), return_counts=True)
    return Coreset(unique, counts.astype(np.int64), np.ones(len(unique), dtype=bool))


def coalesce(state: Coreset) -> Coreset:
    """Sort and combine equal locations without inventing atom purity."""
    if len(state.values) == 0:
        return state
    order = np.argsort(state.values, kind="stable")
    values = state.values[order]
    weights = state.weights[order]
    pure = state.pure[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return Coreset(
        values[starts],
        np.add.reduceat(weights, starts),
        np.logical_and.reduceat(pure, starts),
    )


def union(left: Coreset, right: Coreset) -> Coreset:
    return coalesce(
        Coreset(
            np.r_[left.values, right.values],
            np.r_[left.weights, right.weights],
            np.r_[left.pure, right.pure],
        )
    )


def arcsine_cuts(knots: int) -> np.ndarray:
    # q_j = sin^2(pi*j/(2K)): small tail cells, larger central cells.
    j = np.arange(1, knots, dtype=np.float64)
    return np.sin(np.pi * j / (2 * knots)) ** 2


def arcsine_scale(q: np.ndarray | float) -> np.ndarray | float:
    return (2 / np.pi) * np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def compress(state: Coreset, knots: int = KNOTS) -> Coreset:
    state = coalesce(state)
    if len(state.values) <= knots:
        return state

    cumulative = np.cumsum(state.weights)
    total = int(cumulative[-1])
    boundaries: list[int] = []
    for target in arcsine_cuts(knots) * total:
        index = int(np.searchsorted(cumulative, target, side="left"))
        before = 0 if index == 0 else int(cumulative[index - 1])
        after = int(cumulative[index])
        boundary = index if abs(target - before) <= abs(after - target) else index + 1
        if 0 < boundary < len(state.values):
            boundaries.append(boundary)

    boundaries_array = list(np.unique(boundaries))
    groups = list(
        zip(
            [0, *boundaries_array],
            [*boundaries_array, len(state.values)],
        )
    )

    # Cuts swallowed by a large indivisible tie must not waste the remaining
    # state. Repeatedly bisect the splittable group with the largest span in
    # arcsine rank coordinates until the budget is full.
    prefix = np.r_[0, cumulative]
    while len(groups) < knots:
        best: tuple[float, int, int] | None = None
        for group_index, (start, end) in enumerate(groups):
            if end - start <= 1:
                continue
            left = float(arcsine_scale(prefix[start] / total))
            right = float(arcsine_scale(prefix[end] / total))
            midpoint = (left + right) / 2
            candidates = np.arange(start + 1, end)
            positions = arcsine_scale(prefix[candidates] / total)
            split = int(candidates[np.argmin(np.abs(positions - midpoint))])
            candidate = (right - left, group_index, split)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        _, group_index, split = best
        start, end = groups[group_index]
        groups[group_index : group_index + 1] = [(start, split), (split, end)]

    output_values: list[np.float32] = []
    output_weights: list[int] = []
    output_pure: list[bool] = []
    for start, end in groups:
        weights = state.weights[start:end]
        mass = int(np.sum(weights))
        # The f32 cast is intentional: it simulates the persistent representation.
        representative = np.float32(np.average(state.values[start:end], weights=weights))
        output_values.append(representative)
        output_weights.append(mass)
        output_pure.append(end == start + 1 and bool(state.pure[start]))

    return coalesce(
        Coreset(
            np.asarray(output_values, dtype=np.float32),
            np.asarray(output_weights, dtype=np.int64),
            np.asarray(output_pure, dtype=bool),
        )
    )


def build_stream(values: np.ndarray) -> Coreset:
    state = empty()
    for start in range(0, len(values), BATCH_ROWS):
        state = compress(union(state, exact_runs(values[start : start + BATCH_ROWS])))
    return state


def build_offline(values: np.ndarray) -> Coreset:
    return compress(exact_runs(values))


def quantize_masses(state: Coreset) -> Coreset:
    """Encode masses as u16 probability quanta with prefix-error control."""
    cumulative = np.rint(
        np.cumsum(state.weights, dtype=np.float64)
        / np.sum(state.weights)
        * MASS_QUANTA
    ).astype(np.int64)
    weights = np.diff(np.r_[0, cumulative])
    keep = weights > 0
    return Coreset(state.values[keep], weights[keep], state.pure[keep])


def build_quantized_stream(values: np.ndarray) -> Coreset:
    """64 f32 values + 64 u16 masses + masks/count = 400 bytes."""
    state = empty()
    seen = 0
    for start in range(0, len(values), BATCH_ROWS):
        batch_values = values[start : start + BATCH_ROWS]
        batch = exact_runs(batch_values)
        if seen:
            # Both sides use the common denominator MASS_QUANTA. The old
            # probability masses are scaled by their true historical count.
            old = Coreset(state.values, state.weights * seen, state.pure)
            new = Coreset(
                batch.values,
                batch.weights * MASS_QUANTA,
                batch.pure,
            )
            merged = union(old, new)
        else:
            merged = batch
        state = quantize_masses(compress(merged, QUANTIZED_KNOTS))
        seen += len(batch_values)
    return state


def quantiles(state: Coreset, qs: np.ndarray = QS) -> np.ndarray:
    """Decode a monotone quantile curve while retaining known horizontal ties."""
    cumulative = np.cumsum(state.weights)
    total = int(cumulative[-1])
    before = np.r_[0, cumulative[:-1]]
    anchor_q: list[float] = [0.0]
    anchor_x: list[float] = [float(state.values[0])]
    for value, pure, left, right in zip(
        state.values, state.pure, before, cumulative
    ):
        if pure:
            anchor_q.extend([left / total, right / total])
            anchor_x.extend([float(value), float(value)])
        else:
            anchor_q.append((left + right) / (2 * total))
            anchor_x.append(float(value))
    anchor_q.append(1.0)
    anchor_x.append(float(state.values[-1]))
    # Duplicate q anchors encode a vertical transition between adjacent atoms.
    return np.interp(qs, np.asarray(anchor_q), np.asarray(anchor_x))


def valid_rank_errors(data: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    ordered = np.sort(data)
    left = np.searchsorted(ordered, estimates, side="left") / len(ordered)
    right = np.searchsorted(ordered, estimates, side="right") / len(ordered)
    return np.maximum(np.maximum(left - QS, QS - right), 0)


def datasets(seed: int, n: int = 32_768) -> dict[str, np.ndarray]:
    """Generate tied, smooth, multimodal, and identical-multiset order cases."""
    rng = np.random.default_rng(seed)
    uniform = rng.uniform(-1, 1, n).astype(np.float32)
    normal = rng.normal(size=n).astype(np.float32)

    left = rng.normal(-2, .3, n // 2)
    right = rng.normal(2, .7, n - n // 2)
    mixture = np.r_[left, right].astype(np.float32)
    permutation = rng.permutation(n)

    upper_atom = normal.copy()
    upper_atom[rng.random(n) < .01] = np.float32(20)
    half_zero = normal.copy()
    half_zero[rng.random(n) < .5] = np.float32(0)
    five_uniform = rng.random(n)
    five_atoms = rng.normal(size=n)
    for lo, hi, value in ((0, .12, -5), (.12, .24, -1), (.24, .36, 0), (.36, .48, 2), (.48, .60, 8)):
        mask = (five_uniform >= lo) & (five_uniform < hi)
        five_atoms[mask] = value

    output: dict[str, np.ndarray] = {
        "uniform-random": uniform,
        "uniform-ascending": np.sort(uniform),
        "uniform-descending": np.sort(uniform)[::-1],
        "normal": normal,
        "lognormal-1": rng.lognormal(0, 1, n).astype(np.float32),
        "lognormal-2": rng.lognormal(0, 2, n).astype(np.float32),
        "student-t2": rng.standard_t(2, n).astype(np.float32),
        "mixture-random": mixture[permutation],
        "mixture-blocked": mixture,
        "one-percent-atom": upper_atom,
        "half-zero": half_zero,
        "five-atoms": five_atoms.astype(np.float32),
        "constant": np.full(n, np.float32(3.25)),
    }
    quantization_source = rng.normal(size=n)
    for levels in (8, 16, 32, 64, 256):
        edges = np.quantile(quantization_source, np.linspace(0, 1, levels + 1))
        output[f"quantized-{levels}"] = np.searchsorted(
            edges[1:-1], quantization_source
        ).astype(np.float32)
    return output


def rust_results(
    data: dict[tuple[int, str], np.ndarray], temporary: Path
) -> dict[tuple[int, str, str], tuple[float, float, float]]:
    data_dir = temporary / "data"
    crate_dir = temporary / "baseline"
    source_dir = crate_dir / "src"
    data_dir.mkdir()
    source_dir.mkdir(parents=True)
    identifiers: list[tuple[int, str, str]] = []
    for index, ((seed, name), values) in enumerate(data.items()):
        filename = f"case-{index}.bin"
        (data_dir / filename).write_bytes(values.astype("<f4").tobytes())
        identifiers.append((seed, name, filename))

    package_path = (ROOT / "monatq").as_posix()
    (crate_dir / "Cargo.toml").write_text(
        "[package]\nname='transport-coreset-baseline'\nversion='0.1.0'\n"
        f"edition='2024'\n[dependencies]\nmonatq={{path='{package_path}'}}\n"
    )
    cases = ",".join(
        f'({seed}u64,"{name}","{filename}")'
        for seed, name, filename in identifiers
    )
    qs = ",".join(f"{value:.9}f32" for value in QS)
    rust_source = r'''
use monatq::{DigestKernel, QuantileSpine, TDigest, TensorDigest};
use std::{env, fs};

fn valid_error(sorted: &[f32], x: f32, q: f32) -> f64 {
    let left = sorted.partition_point(|v| *v < x) as f64 / sorted.len() as f64;
    let right = sorted.partition_point(|v| *v <= x) as f64 / sorted.len() as f64;
    (left - q as f64).max(q as f64 - right).max(0.0)
}

fn run<K: DigestKernel<f32>>(seed: u64, name: &str, data: &[f32], qs: &[f32]) {
    let mut digest = TensorDigest::<f32, K>::new(&[1]);
    for &value in data { digest.update(&[value]); }
    let estimates: Vec<f32> = digest.quantiles(qs).into_iter().map(|v| v[0]).collect();
    let mut ordered = data.to_vec();
    ordered.sort_by(f32::total_cmp);
    let errors: Vec<f64> = estimates.iter().zip(qs)
        .map(|(&x, &q)| valid_error(&ordered, x, q)).collect();
    let backend = std::any::type_name::<K>().rsplit("::").next().unwrap();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let maximum = errors.iter().copied().fold(0.0, f64::max);
    let tail = errors.iter().zip(qs).filter(|(_, q)| **q <= 0.01 || **q >= 0.99)
        .map(|(e, _)| *e).fold(0.0, f64::max);
    println!("{seed},{name},{backend},{mean:.8},{maximum:.8},{tail:.8}");
}

fn main() {
    let root = env::args().nth(1).unwrap();
    let qs = [QUANTILES];
    for (seed, name, filename) in [CASES] {
        let bytes = fs::read(format!("{root}/{filename}")).unwrap();
        let data: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        run::<TDigest>(seed, name, &data, &qs);
        run::<QuantileSpine>(seed, name, &data, &qs);
    }
}
'''.replace("QUANTILES", qs).replace("CASES", cases)
    (source_dir / "main.rs").write_text(rust_source)
    process = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--", str(data_dir)],
        cwd=crate_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    output: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    for row in csv.reader(process.stdout.splitlines()):
        seed, name, backend, mean, maximum, tail = row
        output[int(seed), name, backend] = (float(mean), float(maximum), float(tail))
    return output


def main() -> None:
    all_data = {
        (seed, name): values
        for seed in (44, 55)
        for name, values in datasets(seed).items()
    }
    with tempfile.TemporaryDirectory(prefix="transport-coreset-") as directory:
        baselines = rust_results(all_data, Path(directory))

    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["seed", "workload", "backend", "mean_valid_rank_error", "max_valid_rank_error", "tail_max", "knots"]
    )
    summary: dict[str, list[tuple[float, float, float]]] = {
        "RTC-48-stream": [],
        "RTC-64-u16-stream": [],
        "RTC-48-offline": [],
        "TDigest": [],
        "QuantileSpine": [],
    }
    for (seed, name), values in all_data.items():
        for label, state in (
            ("RTC-48-stream", build_stream(values)),
            ("RTC-64-u16-stream", build_quantized_stream(values)),
            ("RTC-48-offline", build_offline(values)),
        ):
            errors = valid_rank_errors(values, quantiles(state))
            metrics = (float(errors.mean()), float(errors.max()), float(errors[TAIL].max()))
            summary[label].append(metrics)
            writer.writerow(
                [seed, name, label, *(f"{value:.8f}" for value in metrics), len(state.values)]
            )
        for backend in ("TDigest", "QuantileSpine"):
            metrics = baselines[seed, name, backend]
            summary[backend].append(metrics)
            writer.writerow([seed, name, backend, *(f"{value:.8f}" for value in metrics), ""])

    writer.writerow([])
    writer.writerow(["SUMMARY", "backend", "mean_of_means", "median_mean", "worst_max", "worst_tail"])
    for backend, rows in summary.items():
        values = np.asarray(rows)
        writer.writerow(
            [
                "SUMMARY",
                backend,
                f"{values[:, 0].mean():.8f}",
                f"{np.median(values[:, 0]):.8f}",
                f"{values[:, 1].max():.8f}",
                f"{values[:, 2].max():.8f}",
            ]
        )


if __name__ == "__main__":
    main()

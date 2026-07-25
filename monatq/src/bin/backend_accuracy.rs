use monatq::dev_support::{BACKENDS, Backend};
use statrs::distribution::{ContinuousCDF, LogNormal, Normal, Uniform};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

/// Process-wide allocator instrumentation for this report binary.
///
/// The counters track allocator-requested bytes rather than estimating the sizes
/// of backend fields. Because Rayon allocations happen on worker threads, the
/// instrument must be global rather than thread-local.
struct TrackingAllocator;

static LIVE_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn record_allocation(bytes: usize) {
    let live = LIVE_HEAP_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_HEAP_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn record_deallocation(bytes: usize) {
    LIVE_HEAP_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        record_deallocation(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let resized = unsafe { System.realloc(pointer, layout, new_size) };
        if !resized.is_null() {
            if new_size >= layout.size() {
                record_allocation(new_size - layout.size());
            } else {
                record_deallocation(layout.size() - new_size);
            }
        }
        resized
    }
}

#[derive(Clone, Copy, Debug)]
struct HeapMeasurement {
    live_bytes: usize,
    peak_bytes: usize,
}

fn begin_heap_measurement() -> usize {
    let baseline = LIVE_HEAP_BYTES.load(Ordering::Relaxed);
    PEAK_HEAP_BYTES.store(baseline, Ordering::Relaxed);
    baseline
}

fn finish_heap_measurement(baseline: usize) -> HeapMeasurement {
    HeapMeasurement {
        live_bytes: LIVE_HEAP_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(baseline),
        peak_bytes: PEAK_HEAP_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(baseline),
    }
}

fn warm_parallel_runtime() {
    // Initialize Rayon's global pool before opening a backend measurement region.
    // Otherwise one-time runtime allocations would be charged to the first backend.
    rayon::broadcast(|_| std::hint::black_box(Vec::<u8>::with_capacity(64)));
}

fn warm_backend_paths() {
    // Exercise each backend once so global lazy initialization is not mistaken for
    // memory owned by the first measured digest.
    for &backend in BACKENDS {
        let mut digest = backend.create(&[1]);
        for index in 0..512 {
            digest.update(&[(index as f32 * 0.125).sin()]);
        }
        digest.flush();
        std::hint::black_box(digest.quantile(0.5));
    }
}

#[derive(Debug)]
struct Accuracy {
    mean_rank_error: f64,
    max_rank_error: f64,
    worst_quantile: f32,
    live_heap_bytes: usize,
    peak_heap_bytes: usize,
}

fn xorshift32(state: &mut u32) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state as f64) + 0.5) / (u32::MAX as f64 + 1.0)
}

fn rank_interval_error(sorted: &[f32], estimate: f32, q: f32) -> f64 {
    let lower = sorted.partition_point(|&value| value < estimate) as f64 / sorted.len() as f64;
    let upper = sorted.partition_point(|&value| value <= estimate) as f64 / sorted.len() as f64;
    let q = q as f64;
    if q < lower {
        lower - q
    } else if q > upper {
        q - upper
    } else {
        0.0
    }
}

fn measure(backend: Backend, data: &[f32], numel: usize, quantiles: &[f32]) -> Accuracy {
    let mut truth = (0..numel)
        .map(|_| Vec::with_capacity(data.len() / numel))
        .collect::<Vec<_>>();
    for sample in data.chunks_exact(numel) {
        for (position, &value) in sample.iter().enumerate() {
            truth[position].push(value);
        }
    }
    for values in &mut truth {
        values.sort_unstable_by(f32::total_cmp);
    }

    // Inputs and exact truth are fully allocated before this region. Query outputs
    // are allocated after it. The measurement therefore covers only construction,
    // update, flush, retained backend heap, and transient ingestion workspace.
    let baseline = begin_heap_measurement();
    let mut digest = backend.create(&[numel]);
    for sample in data.chunks_exact(numel) {
        digest.update(sample);
    }
    digest.flush();
    let heap = finish_heap_measurement(baseline);

    let estimates = digest.quantiles(quantiles);
    let mut sum = 0.0;
    let mut max = 0.0f64;
    let mut worst_quantile = 0.0;
    for (q_index, &q) in quantiles.iter().enumerate() {
        for (position, sorted) in truth.iter().enumerate() {
            let error = rank_interval_error(sorted, estimates[q_index][position], q);
            sum += error;
            if error > max {
                max = error;
                worst_quantile = q;
            }
        }
    }

    // Measure retained backend heap by observing what its destructor actually
    // releases. This excludes any process-global lazy allocation initialized in
    // the region. Remove the same residual from the recorded ingestion peak.
    let before_drop = LIVE_HEAP_BYTES.load(Ordering::Relaxed);
    drop(digest);
    let after_drop = LIVE_HEAP_BYTES.load(Ordering::Relaxed);
    let live_heap_bytes = before_drop.saturating_sub(after_drop);
    let external_retained = heap.live_bytes.saturating_sub(live_heap_bytes);

    Accuracy {
        mean_rank_error: sum / (quantiles.len() * numel) as f64,
        max_rank_error: max,
        worst_quantile,
        live_heap_bytes,
        peak_heap_bytes: heap.peak_bytes.saturating_sub(external_retained),
    }
}

fn print_table_header(title: &str) {
    println!("\n{title}");
    println!("{}", "─".repeat(124));
    println!(
        "{:<28}  {:<16}  {:>15}  {:>27}  {:>12}  {:>12}",
        "Workload", "Backend", "Mean rank err", "Max rank err (worst q)", "Live heap", "Peak heap"
    );
    println!("{}", "─".repeat(124));
}

fn hsv_to_rgb(hue: f64, saturation: f64, value: f64) -> (u8, u8, u8) {
    let chroma = value * saturation;
    let hue_sector = hue / 60.0;
    let secondary = chroma * (1.0 - (hue_sector.rem_euclid(2.0) - 1.0).abs());
    let (red, green, blue) = match hue_sector as u8 {
        0 => (chroma, secondary, 0.0),
        1 => (secondary, chroma, 0.0),
        2 => (0.0, chroma, secondary),
        3 => (0.0, secondary, chroma),
        4 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };
    let offset = value - chroma;
    let channel = |component: f64| ((component + offset) * 255.0).round() as u8;
    (channel(red), channel(green), channel(blue))
}

fn performance_score(value: f64, best: f64, worst: f64) -> f64 {
    if best == worst {
        1.0
    } else {
        ((worst - value) / (worst - best)).clamp(0.0, 1.0)
    }
}

fn color_metric(value: String, score: f64) -> String {
    // HSV hue 0° is red, 60° is yellow, and 120° is green.
    let (red, green, blue) = hsv_to_rgb(120.0 * score, 1.0, 1.0);
    let bold = if score == 1.0 { "1;" } else { "" };
    format!("\x1b[{bold}38;2;{red};{green};{blue}m{value}\x1b[0m")
}

fn format_bytes(bytes: usize) -> String {
    format!("{bytes} B")
}

fn report(name: &str, data: &[f32], numel: usize, quantiles: &[f32]) {
    let results = BACKENDS
        .iter()
        .map(|&backend| (backend, measure(backend, data, numel, quantiles)))
        .collect::<Vec<_>>();
    let best_mean = results
        .iter()
        .map(|(_, accuracy)| accuracy.mean_rank_error)
        .fold(f64::INFINITY, f64::min);
    let best_max = results
        .iter()
        .map(|(_, accuracy)| accuracy.max_rank_error)
        .fold(f64::INFINITY, f64::min);
    let best_live_heap = results
        .iter()
        .map(|(_, accuracy)| accuracy.live_heap_bytes)
        .min()
        .unwrap_or(0);
    let best_peak_heap = results
        .iter()
        .map(|(_, accuracy)| accuracy.peak_heap_bytes)
        .min()
        .unwrap_or(0);
    let worst_mean = results
        .iter()
        .map(|(_, accuracy)| accuracy.mean_rank_error)
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_max = results
        .iter()
        .map(|(_, accuracy)| accuracy.max_rank_error)
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_live_heap = results
        .iter()
        .map(|(_, accuracy)| accuracy.live_heap_bytes)
        .max()
        .unwrap_or(0);
    let worst_peak_heap = results
        .iter()
        .map(|(_, accuracy)| accuracy.peak_heap_bytes)
        .max()
        .unwrap_or(0);

    for (backend, accuracy) in results {
        let backend_name = format!("{backend:?}");
        let mean = color_metric(
            format!("{:>15.8}", accuracy.mean_rank_error),
            performance_score(accuracy.mean_rank_error, best_mean, worst_mean),
        );
        let max = color_metric(
            format!(
                "{:>15.8} (q={:.5})",
                accuracy.max_rank_error, accuracy.worst_quantile
            ),
            performance_score(accuracy.max_rank_error, best_max, worst_max),
        );
        let live_heap = color_metric(
            format!("{:>12}", format_bytes(accuracy.live_heap_bytes)),
            performance_score(
                accuracy.live_heap_bytes as f64,
                best_live_heap as f64,
                worst_live_heap as f64,
            ),
        );
        let peak_heap = color_metric(
            format!("{:>12}", format_bytes(accuracy.peak_heap_bytes)),
            performance_score(
                accuracy.peak_heap_bytes as f64,
                best_peak_heap as f64,
                worst_peak_heap as f64,
            ),
        );
        println!("{name:<28}  {backend_name:<16}  {mean}  {max}  {live_heap}  {peak_heap}");
    }
    println!();
}

fn laplace_sample(state: &mut u32) -> f32 {
    let probability = xorshift32(state);
    if probability < 0.5 {
        (2.0 * probability).ln() as f32
    } else {
        (-(2.0 * (1.0 - probability)).ln()) as f32
    }
}

fn quantize(value: f32, minimum: f32, maximum: f32, levels: usize) -> f32 {
    let step = (maximum - minimum) / (levels - 1) as f32;
    minimum + ((value.clamp(minimum, maximum) - minimum) / step).round() * step
}

fn regular_reports() {
    print_table_header("Representative distributions (32 tensor positions)");

    const N: usize = 100_000;
    const NUMEL: usize = 32;
    const QUANTILES: &[f32] = &[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(-2.0, 3.0).unwrap();
    let lognormal = LogNormal::new(0.0, 1.0).unwrap();

    for (index, name) in [
        "normal",
        "uniform",
        "lognormal",
        "exponential",
        "laplace",
        "overlapping-bimodal",
        "32-level-normal",
        "50%-zeros",
        "95%-zero-activations",
    ]
    .into_iter()
    .enumerate()
    {
        let mut state = 0x6a09_e667 ^ (index as u32).wrapping_mul(0x9e37_79b9);
        let data = (0..N * NUMEL)
            .map(|_| match name {
                "normal" => normal.inverse_cdf(xorshift32(&mut state)) as f32,
                "uniform" => uniform.inverse_cdf(xorshift32(&mut state)) as f32,
                "lognormal" => lognormal.inverse_cdf(xorshift32(&mut state)) as f32,
                "exponential" => -xorshift32(&mut state).ln() as f32,
                "laplace" => laplace_sample(&mut state),
                "overlapping-bimodal" => {
                    let center = if xorshift32(&mut state) < 0.35 {
                        -1.0
                    } else {
                        1.0
                    };
                    center + normal.inverse_cdf(xorshift32(&mut state)) as f32
                }
                "32-level-normal" => quantize(
                    normal.inverse_cdf(xorshift32(&mut state)) as f32,
                    -4.0,
                    4.0,
                    32,
                ),
                "50%-zeros" if xorshift32(&mut state) < 0.5 => 0.0,
                "50%-zeros" => normal.inverse_cdf(xorshift32(&mut state)) as f32,
                "95%-zero-activations" if xorshift32(&mut state) < 0.95 => 0.0,
                "95%-zero-activations" => lognormal.inverse_cdf(xorshift32(&mut state)) as f32,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        report(name, &data, NUMEL, QUANTILES);
    }

    // Real tensors rarely have identically distributed positions. Mix shapes,
    // scales, ties, and skew across positions in the same digest.
    let mut state = 0x510e_527f;
    let heterogeneous = (0..N * NUMEL)
        .map(|index| {
            let position = index % NUMEL;
            match position % 6 {
                0 => normal.inverse_cdf(xorshift32(&mut state)) as f32,
                1 => (position as f32 + 1.0) * xorshift32(&mut state) as f32,
                2 => -xorshift32(&mut state).ln() as f32,
                3 => laplace_sample(&mut state) * (1.0 + position as f32 / 8.0),
                4 => quantize(
                    normal.inverse_cdf(xorshift32(&mut state)) as f32,
                    -3.0,
                    3.0,
                    16,
                ),
                _ if xorshift32(&mut state) < 0.8 => 0.0,
                _ => lognormal.inverse_cdf(xorshift32(&mut state)) as f32,
            }
        })
        .collect::<Vec<_>>();
    report("heterogeneous-tensor", &heterogeneous, NUMEL, QUANTILES);
}

fn coherent_stripes(n: usize, batch_len: usize, bands: usize, repeats: usize) -> Vec<f32> {
    (0..n)
        .map(|index| {
            let batch = index / batch_len;
            let band = (batch / repeats) % bands;
            let within_band = index % batch_len;
            (band as f32 + (within_band as f32 + 0.5) / batch_len as f32) / bands as f32
        })
        .collect()
}

fn blocked_two_mode(n: usize, run_len: usize) -> Vec<f32> {
    let samples_per_mode = n / 2;
    (0..n)
        .map(|index| {
            let cycle = index / (2 * run_len);
            let within_run = index % run_len;
            let value = (cycle * run_len + within_run) as f32 / (samples_per_mode - 1) as f32;
            if index % (2 * run_len) < run_len {
                value
            } else {
                10.0 + value
            }
        })
        .collect()
}

fn batch_edge_runs(n: usize) -> Vec<f32> {
    let run_lengths = [255, 256, 257];
    let mut data = Vec::with_capacity(n);
    let mut run = 0;
    while data.len() < n {
        let run_len = run_lengths[run % run_lengths.len()];
        let base = if run % 2 == 0 { -10.0 } else { 10.0 };
        for within_run in 0..run_len.min(n - data.len()) {
            data.push(base + within_run as f32 / run_len as f32);
        }
        run += 1;
    }
    data
}

fn repeated_regime_shifts(n: usize, segment_len: usize) -> Vec<f32> {
    let centers = [-1_000_000.0, 0.0, 1_000_000.0, 10.0];
    (0..n)
        .map(|index| {
            let segment = index / segment_len;
            let phase = (index % segment_len) as f32 / segment_len as f32;
            centers[segment % centers.len()] + phase
        })
        .collect()
}

fn adversarial_quantiles() -> Vec<f32> {
    let mut quantiles = (1..1_000)
        .map(|index| index as f32 / 1_000.0)
        .collect::<Vec<_>>();
    quantiles.extend([0.0001, 0.0005, 0.9995, 0.9999]);
    quantiles.sort_unstable_by(f32::total_cmp);
    quantiles.dedup();
    quantiles
}

fn adversarial_reports() {
    print_table_header("Adversarial streams (1 tensor position)");

    const N: usize = 65_536;
    const BATCH_LEN: usize = 256;
    let quantiles = adversarial_quantiles();

    let mut state = 0x6a09_e667;
    let shuffled = (0..N)
        .map(|_| xorshift32(&mut state) as f32)
        .collect::<Vec<_>>();
    report("shuffled-uniform", &shuffled, 1, &quantiles);

    let mut state = 0x0001_5ba4;
    let rare_upper_atom = (0..N)
        .map(|_| {
            if xorshift32(&mut state) < 0.025 {
                10.0
            } else {
                xorshift32(&mut state) as f32
            }
        })
        .collect::<Vec<_>>();
    report("rare-upper-atom", &rare_upper_atom, 1, &quantiles);

    let mut state = 0xbb67_ae85;
    let needle_outliers = (0..N)
        .map(|_| {
            let draw = xorshift32(&mut state);
            if draw < 0.0005 {
                1.0e30
            } else if draw > 0.9995 {
                -1.0e30
            } else {
                xorshift32(&mut state) as f32
            }
        })
        .collect::<Vec<_>>();
    report("0.1%-needle-outliers", &needle_outliers, 1, &quantiles);

    let many_atoms = (0..N)
        .map(|index| ((index * 73) % 128) as f32)
        .collect::<Vec<_>>();
    report("128-shuffled-atoms", &many_atoms, 1, &quantiles);

    let mut state = 0x3c6e_f372;
    let extreme_dynamic_range = (0..N)
        .map(|index| {
            let magnitude = match index % 3 {
                0 => 1.0e-30,
                1 => 1.0,
                _ => 1.0e30,
            };
            let sign = if index % 2 == 0 { -1.0 } else { 1.0 };
            sign * magnitude * xorshift32(&mut state) as f32
        })
        .collect::<Vec<_>>();
    report(
        "extreme-dynamic-range",
        &extreme_dynamic_range,
        1,
        &quantiles,
    );

    report("batch-edge-runs", &batch_edge_runs(N), 1, &quantiles);
    report(
        "repeated-regime-shifts",
        &repeated_regime_shifts(N, BATCH_LEN),
        1,
        &quantiles,
    );

    let alternating_extremes = (0..N)
        .map(|index| {
            let offset = index as f32 / N as f32;
            if index % 2 == 0 {
                -1.0e20 + offset * 1.0e18
            } else {
                1.0e20 + offset * 1.0e18
            }
        })
        .collect::<Vec<_>>();
    report("alternating-extremes", &alternating_extremes, 1, &quantiles);

    let ascending = (0..N)
        .map(|index| index as f32 / (N - 1) as f32)
        .collect::<Vec<_>>();
    let descending = ascending.iter().copied().rev().collect::<Vec<_>>();
    for (name, data) in [
        (
            "single-striped-uniform",
            coherent_stripes(N, BATCH_LEN, 16, 1),
        ),
        (
            "repeated-striped-uniform",
            coherent_stripes(N, BATCH_LEN, 16, 4),
        ),
        ("ascending-uniform", ascending),
        ("descending-uniform", descending),
        ("blocked-two-mode", blocked_two_mode(N, BATCH_LEN)),
    ] {
        report(name, &data, 1, &quantiles);
    }
}

fn main() {
    warm_parallel_runtime();
    warm_backend_paths();
    println!("TensorDigest backend accuracy report");
    println!("Lower errors and memory use are better.");
    println!("Metric colors interpolate in HSV from red (worst) through yellow to green (best).");
    println!(
        "Heap bytes are measured by the instrumented global allocator, not calculated from backend fields."
    );
    println!(
        "Live is retained after flush; peak covers backend construction, update, flush, and ingestion workspace."
    );
    println!("Input data, exact truth, and query outputs are excluded.");
    regular_reports();
    adversarial_reports();
}

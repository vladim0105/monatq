use monatq::dev_support::{BACKENDS, Backend};
use statrs::distribution::{ContinuousCDF, LogNormal, Normal, Uniform};

#[derive(Debug)]
struct Accuracy {
    mean_rank_error: f64,
    max_rank_error: f64,
    worst_quantile: f32,
    memory_bytes: usize,
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
    let mut digest = backend.create(&[numel]);
    for sample in data.chunks_exact(numel) {
        digest.update(sample);
        for (position, &value) in sample.iter().enumerate() {
            truth[position].push(value);
        }
    }
    digest.flush();
    for values in &mut truth {
        values.sort_unstable_by(f32::total_cmp);
    }

    let memory_bytes = digest.allocated_memory_bytes();
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
    Accuracy {
        mean_rank_error: sum / (quantiles.len() * numel) as f64,
        max_rank_error: max,
        worst_quantile,
        memory_bytes,
    }
}

fn print_table_header(title: &str) {
    println!("\n{title}");
    println!("{}", "─".repeat(104));
    println!(
        "{:<28}  {:<16}  {:>15}  {:>25}  {:>12}",
        "Workload", "Backend", "Mean rank err", "Max rank err (worst q)", "Memory"
    );
    println!("{}", "─".repeat(104));
}

fn bold(value: String, is_best: bool) -> String {
    if is_best {
        format!("\x1b[1m{value}\x1b[0m")
    } else {
        value
    }
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
    let best_memory = results
        .iter()
        .map(|(_, accuracy)| accuracy.memory_bytes)
        .min()
        .unwrap_or(0);

    for (backend, accuracy) in results {
        let backend_name = format!("{backend:?}");
        let mean = bold(
            format!("{:>15.8}", accuracy.mean_rank_error),
            accuracy.mean_rank_error == best_mean,
        );
        let max = bold(
            format!(
                "{:>15.8} (q={:.3})",
                accuracy.max_rank_error, accuracy.worst_quantile
            ),
            accuracy.max_rank_error == best_max,
        );
        let memory = bold(
            format!("{:>9.1} KiB", accuracy.memory_bytes as f64 / 1024.0),
            accuracy.memory_bytes == best_memory,
        );
        println!("{name:<28}  {backend_name:<16}  {mean}  {max}  {memory}");
    }
    println!();
}

fn regular_reports() {
    print_table_header("Representative distributions");

    const N: usize = 100_000;
    const NUMEL: usize = 32;
    const QUANTILES: &[f32] = &[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(-2.0, 3.0).unwrap();
    let lognormal = LogNormal::new(0.0, 1.0).unwrap();

    for (index, name) in ["normal", "uniform", "lognormal", "50%-zeros"]
        .into_iter()
        .enumerate()
    {
        let mut state = 0x6a09_e667 ^ (index as u32).wrapping_mul(0x9e37_79b9);
        let data = (0..N * NUMEL)
            .map(|_| match name {
                "normal" => normal.inverse_cdf(xorshift32(&mut state)) as f32,
                "uniform" => uniform.inverse_cdf(xorshift32(&mut state)) as f32,
                "lognormal" => lognormal.inverse_cdf(xorshift32(&mut state)) as f32,
                _ if xorshift32(&mut state) < 0.5 => 0.0,
                _ => normal.inverse_cdf(xorshift32(&mut state)) as f32,
            })
            .collect::<Vec<_>>();
        report(name, &data, NUMEL, QUANTILES);
    }
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

fn adversarial_reports() {
    print_table_header("Adversarial streams");

    const N: usize = 65_536;
    const BATCH_LEN: usize = 256;
    let quantiles = (1..1_000)
        .map(|index| index as f32 / 1_000.0)
        .collect::<Vec<_>>();

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
    println!("TensorDigest backend accuracy report");
    println!("Lower errors and memory use are better.");
    regular_reports();
    adversarial_reports();
}

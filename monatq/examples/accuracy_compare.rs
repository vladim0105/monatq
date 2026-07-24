use monatq::{QuantileSpine, TensorDigest};
use statrs::distribution::{ContinuousCDF, LogNormal, Normal, Uniform};
use std::mem::size_of;

const QUANTILES: [f32; 9] = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999];

#[derive(Clone, Copy, Debug)]
enum Workload {
    Normal,
    Uniform,
    Lognormal,
    HalfZeros,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Uniform => "uniform",
            Self::Lognormal => "lognormal",
            Self::HalfZeros => "50%-zeros",
        }
    }
}

fn xorshift32(state: &mut u32) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state as f64) + 0.5) / (u32::MAX as f64 + 1.0)
}

fn draw(
    workload: Workload,
    state: &mut u32,
    normal: &Normal,
    uniform: &Uniform,
    lognormal: &LogNormal,
) -> f32 {
    match workload {
        Workload::Normal => normal.inverse_cdf(xorshift32(state)) as f32,
        Workload::Uniform => uniform.inverse_cdf(xorshift32(state)) as f32,
        Workload::Lognormal => lognormal.inverse_cdf(xorshift32(state)) as f32,
        Workload::HalfZeros => {
            if xorshift32(state) < 0.5 {
                0.0
            } else {
                normal.inverse_cdf(xorshift32(state)) as f32
            }
        }
    }
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

fn compare_workload(
    workload: Workload,
    n: usize,
    numel: usize,
) -> (f64, f64, f64, f64, [f64; QUANTILES.len()]) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(-2.0, 3.0).unwrap();
    let lognormal = LogNormal::new(0.0, 1.0).unwrap();
    let mut state = 0x6a09_e667 ^ (workload as u32).wrapping_mul(0x9e37_79b9);
    let mut truth = (0..numel)
        .map(|_| Vec::with_capacity(n))
        .collect::<Vec<_>>();
    let mut row = vec![0.0f32; numel];
    let mut spine = QuantileSpine::new(&[numel]);
    let mut digest = TensorDigest::new(&[numel], 100);

    for _ in 0..n {
        for (position, value) in row.iter_mut().enumerate() {
            *value = draw(workload, &mut state, &normal, &uniform, &lognormal);
            truth[position].push(*value);
        }
        spine.update(&row);
        digest.update(&row);
    }
    for values in &mut truth {
        values.sort_unstable_by(f32::total_cmp);
    }

    let spine_estimates = spine.quantiles(&QUANTILES);
    let digest_estimates = digest.quantiles(&QUANTILES);
    let mut spine_sum = 0.0;
    let mut spine_max = 0.0f64;
    let mut digest_sum = 0.0;
    let mut digest_max = 0.0f64;
    let mut spine_max_by_quantile = [0.0f64; QUANTILES.len()];
    let comparisons = (QUANTILES.len() * numel) as f64;
    for (q_index, &q) in QUANTILES.iter().enumerate() {
        for (position, exact) in truth.iter().enumerate() {
            let spine_error = rank_interval_error(exact, spine_estimates[q_index][position], q);
            let digest_error = rank_interval_error(exact, digest_estimates[q_index][position], q);
            spine_sum += spine_error;
            spine_max = spine_max.max(spine_error);
            spine_max_by_quantile[q_index] = spine_max_by_quantile[q_index].max(spine_error);
            digest_sum += digest_error;
            digest_max = digest_max.max(digest_error);
        }
    }

    (
        spine_sum / comparisons,
        spine_max,
        digest_sum / comparisons,
        digest_max,
        spine_max_by_quantile,
    )
}

fn tdigest_state_bytes_per_position<T>() -> usize {
    let max_centroids = 6 * 100 + 10;
    max_centroids * (size_of::<f32>() + size_of::<u32>())
        + size_of::<usize>()
        + size_of::<u32>()
        + 2 * size_of::<T>()
}

fn drift_comparison(numel: usize) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut state = 0xbb67_ae85;
    let mut row = vec![0.0f32; numel];
    let mut spine = QuantileSpine::new(&[numel]);
    let mut digest = TensorDigest::new(&[numel], 100);

    for _ in 0..50_000 {
        for value in &mut row {
            *value = normal.inverse_cdf(xorshift32(&mut state)) as f32;
        }
        spine.update(&row);
        digest.update(&row);
    }

    println!("\ndrift after N(0,1) -> N(4,1): mean absolute median value error");
    println!("post_samples,spine,t-digest");
    let checkpoints = [256usize, 1_024, 4_096, 16_384, 50_000];
    let mut previous = 0;
    for checkpoint in checkpoints {
        for _ in previous..checkpoint {
            for value in &mut row {
                *value = (normal.inverse_cdf(xorshift32(&mut state)) + 4.0) as f32;
            }
            spine.update(&row);
            digest.update(&row);
        }
        let spine_medians = spine.quantile(0.5);
        let digest_medians = digest.quantile(0.5);
        let spine_error = spine_medians
            .iter()
            .map(|value| (value - 4.0).abs() as f64)
            .sum::<f64>()
            / numel as f64;
        let digest_error = digest_medians
            .iter()
            .map(|value| (value - 4.0).abs() as f64)
            .sum::<f64>()
            / numel as f64;
        println!("{checkpoint},{spine_error:.6},{digest_error:.6}");
        previous = checkpoint;
    }
}

fn main() {
    let n = std::env::var("MONATQ_ACCURACY_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100_000);
    let numel = std::env::var("MONATQ_ACCURACY_NUMEL")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);

    println!("N={n}, positions={numel}, quantiles={QUANTILES:?}");
    println!("distribution,spine_mean,spine_max,t-digest_mean,t-digest_max");
    for workload in [
        Workload::Normal,
        Workload::Uniform,
        Workload::Lognormal,
        Workload::HalfZeros,
    ] {
        let (spine_mean, spine_max, digest_mean, digest_max, spine_max_by_quantile) =
            compare_workload(workload, n, numel);
        println!(
            "{},{spine_mean:.8},{spine_max:.8},{digest_mean:.8},{digest_max:.8}",
            workload.name()
        );
        println!(
            "  spine max rank error by q: {}",
            QUANTILES
                .iter()
                .zip(spine_max_by_quantile)
                .map(|(q, error)| format!("{q:.3}={error:.8}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let spine = QuantileSpine::<f32>::new(&[numel]);
    let spine_bytes = spine.state_memory_bytes() / numel;
    let digest_bytes = tdigest_state_bytes_per_position::<f32>();
    println!(
        "\npersistent bytes/position (actual field element sizes): spine={spine_bytes}, t-digest={digest_bytes}, ratio={:.2}x",
        digest_bytes as f64 / spine_bytes as f64
    );

    drift_comparison(numel);
}

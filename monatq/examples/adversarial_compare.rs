use monatq::{QuantileSpine, QuantileSpineConfig, TensorDigest};

// A dense grid is important here: a short list of conventional quantiles can hide a narrow
// failure. Every reported error covers q=.001, .002, ..., .999.
const GRID_STEPS: usize = 1_000;

struct Accuracy {
    estimates: Vec<f32>,
    mean_rank_error: f64,
    max_rank_error: f64,
    worst_quantile: f32,
}

impl Accuracy {
    fn estimate(&self, q: f32) -> f32 {
        let index = (q * GRID_STEPS as f32).round() as usize;
        self.estimates[index.clamp(1, GRID_STEPS - 1) - 1]
    }
}

fn quantile_grid() -> Vec<f32> {
    (1..GRID_STEPS)
        .map(|index| index as f32 / GRID_STEPS as f32)
        .collect()
}

fn xorshift32(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state as f64 + 0.5) / (u32::MAX as f64 + 1.0)) as f32
}

/// Gives tied estimates credit for the full empirical rank interval occupied by the tie.
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

fn summarize(sorted: &[f32], quantiles: &[f32], estimates: Vec<f32>) -> Accuracy {
    let errors = quantiles
        .iter()
        .zip(&estimates)
        .map(|(&q, &estimate)| rank_interval_error(sorted, estimate, q))
        .collect::<Vec<_>>();
    let (worst_index, &max_rank_error) = errors
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .expect("the quantile grid is nonempty");
    Accuracy {
        estimates,
        mean_rank_error: errors.iter().sum::<f64>() / errors.len() as f64,
        max_rank_error,
        worst_quantile: quantiles[worst_index],
    }
}

fn sorted_truth(values: &[f32]) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    sorted
}

fn evaluate_spine(values: &[f32], config: QuantileSpineConfig, quantiles: &[f32]) -> Accuracy {
    let mut spine = QuantileSpine::with_config(&[1], config);
    for &value in values {
        spine.update(&[value]);
    }
    summarize(
        &sorted_truth(values),
        quantiles,
        spine.cell_quantiles(0, quantiles),
    )
}

fn evaluate_digest(values: &[f32], quantiles: &[f32]) -> Accuracy {
    let mut digest = TensorDigest::new(&[1], 100);
    for &value in values {
        digest.update(&[value]);
    }
    summarize(
        &sorted_truth(values),
        quantiles,
        digest.cell_quantiles(0, quantiles),
    )
}

fn print_comparison(name: &str, values: &[f32], quantiles: &[f32]) -> (Accuracy, Accuracy) {
    let spine = evaluate_spine(values, QuantileSpineConfig::default(), quantiles);
    let digest = evaluate_digest(values, quantiles);
    println!(
        "{name},{:.8},{:.8},{:.3},{:.8},{:.8},{:.3},{:.2}",
        spine.mean_rank_error,
        spine.max_rank_error,
        spine.worst_quantile,
        digest.mean_rank_error,
        digest.max_rank_error,
        digest.worst_quantile,
        spine.mean_rank_error / digest.mean_rank_error.max(f64::EPSILON),
    );
    (spine, digest)
}

/// Uniform values presented as narrow, sorted batches. Repeating each batch makes consecutive
/// summaries coherent, so the surprise policy interprets an ordering pattern as regime changes.
fn coherent_stripes(n: usize, batch_len: usize, bands: usize, repeats: usize) -> Vec<f32> {
    assert_eq!(n % (batch_len * bands * repeats), 0);
    (0..n)
        .map(|index| {
            let batch = index / batch_len;
            let band = (batch / repeats) % bands;
            let within_band = index % batch_len;
            (band as f32 + (within_band as f32 + 0.5) / batch_len as f32) / bands as f32
        })
        .collect()
}

/// Equal amounts of U(0,1) and U(10,11), alternating one sorted batch at a time.
fn blocked_two_mode(n: usize, run_len: usize) -> Vec<f32> {
    assert_eq!(n % (2 * run_len), 0);
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

fn main() {
    const N: usize = 65_536;
    const BATCH_LEN: usize = 256;
    let quantiles = quantile_grid();

    // Control: the default spine wins on an ordinary randomly ordered uniform stream.
    let mut state = 0x6a09_e667;
    let shuffled = (0..N).map(|_| xorshift32(&mut state)).collect::<Vec<_>>();

    // Representation stress test. This used to be a sub-threshold-atom failure, but retaining it
    // guards against regressions and demonstrates that the suite is not selected only for wins.
    let mut state = 0x0001_5ba4;
    let rare_upper_atom = (0..N)
        .map(|_| {
            if xorshift32(&mut state) < 0.025 {
                10.0
            } else {
                xorshift32(&mut state)
            }
        })
        .collect::<Vec<_>>();

    // The same globally uniform values under increasingly hostile orderings.
    let single_stripes = coherent_stripes(N, BATCH_LEN, 16, 1);
    let repeated_stripes = coherent_stripes(N, BATCH_LEN, 16, 4);
    let ascending = (0..N)
        .map(|index| index as f32 / (N - 1) as f32)
        .collect::<Vec<_>>();
    let descending = ascending.iter().copied().rev().collect::<Vec<_>>();

    // A separate gap/interpolation stressor. Both sketches find this harder than uniform data,
    // but t-digest has lower mean rank error for this batch-aligned ordering.
    let two_mode = blocked_two_mode(N, BATCH_LEN);

    println!("N={N}, q-grid=.001..=.999, t-digest compression=100");
    println!(
        "workload,spine_mean,spine_max,spine_worst_q,t-digest_mean,t-digest_max,t-digest_worst_q,mean_ratio"
    );
    print_comparison("shuffled-uniform", &shuffled, &quantiles);
    print_comparison("rare-upper-atom", &rare_upper_atom, &quantiles);
    print_comparison("single-striped-uniform", &single_stripes, &quantiles);
    let (repeated_spine, repeated_digest) =
        print_comparison("repeated-striped-uniform", &repeated_stripes, &quantiles);
    print_comparison("ascending-uniform", &ascending, &quantiles);
    print_comparison("descending-uniform", &descending, &quantiles);
    print_comparison("blocked-two-mode", &two_mode, &quantiles);

    // These knobs now affect only the recent view. The global result is intentionally identical:
    // its count-only gain is an invariant rather than a configuration choice.
    let no_adaptation = QuantileSpineConfig {
        n_max: u64::MAX,
        gain_c: 0.0,
        restart_crossings: 0,
        ..QuantileSpineConfig::default()
    };
    let repeated_no_adaptation = evaluate_spine(&repeated_stripes, no_adaptation, &quantiles);
    println!(
        "repeated-striped-uniform (global, recent adaptation disabled),{:.8},{:.8},{:.3},-,-,-,-",
        repeated_no_adaptation.mean_rank_error,
        repeated_no_adaptation.max_rank_error,
        repeated_no_adaptation.worst_quantile,
    );

    println!("\nselected estimates (all exact values are approximately q for uniform data):");
    for q in [0.01, 0.5, 0.99] {
        println!(
            "repeated-striped q={q:.2}: spine-global={:.6}, t-digest={:.6}, global-with-recent-disabled={:.6}",
            repeated_spine.estimate(q),
            repeated_digest.estimate(q),
            repeated_no_adaptation.estimate(q),
        );
    }
}

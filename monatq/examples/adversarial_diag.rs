use monatq::TensorDigest;
use monatq::{QuantileSpine, QuantileSpineConfig};

const GRID_STEPS: usize = 1_000;

fn quantile_grid() -> Vec<f32> {
    (1..GRID_STEPS)
        .map(|index| index as f32 / GRID_STEPS as f32)
        .collect()
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

fn eval(values: &[f32], config: QuantileSpineConfig) -> (f64, f64) {
    let quantiles = quantile_grid();
    let mut spine = TensorDigest::<_, QuantileSpine>::with_config(&[1], config);
    for &value in values {
        spine.update(&[value]).unwrap();
    }
    let estimates = spine.cell_quantiles(0, &quantiles).unwrap();
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let errors: Vec<f64> = quantiles
        .iter()
        .zip(&estimates)
        .map(|(&q, &e)| rank_interval_error(&sorted, e, q))
        .collect();
    (
        errors.iter().sum::<f64>() / errors.len() as f64,
        errors.iter().cloned().fold(0.0, f64::max),
    )
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

fn main() {
    const N: usize = 65_536;
    const BATCH_LEN: usize = 256;
    let ascending: Vec<f32> = (0..N).map(|i| i as f32 / (N - 1) as f32).collect();
    let single_stripes = coherent_stripes(N, BATCH_LEN, 16, 1);
    let repeated_stripes = coherent_stripes(N, BATCH_LEN, 16, 4);

    let default = QuantileSpineConfig::default();
    let no_restart = QuantileSpineConfig {
        restart_crossings: 0,
        ..default
    };
    let no_gain = QuantileSpineConfig {
        gain_c: 0.0,
        ..default
    };
    let no_fade = QuantileSpineConfig {
        n_max: u64::MAX,
        ..default
    };
    let no_gain_no_restart = QuantileSpineConfig {
        gain_c: 0.0,
        restart_crossings: 0,
        ..default
    };
    let none = QuantileSpineConfig {
        gain_c: 0.0,
        restart_crossings: 0,
        n_max: u64::MAX,
        ..default
    };

    let configs: [(&str, QuantileSpineConfig); 6] = [
        ("default", default),
        ("no-restart", no_restart),
        ("no-surprise-gain", no_gain),
        ("no-fading", no_fade),
        ("no-gain+no-restart", no_gain_no_restart),
        ("all-off", none),
    ];
    let workloads: [(&str, &[f32]); 3] = [
        ("ascending", &ascending),
        ("single-stripes", &single_stripes),
        ("repeated-stripes", &repeated_stripes),
    ];

    println!("workload,recent-view-config (global invariant),mean_rank_err,max_rank_err");
    for (wname, values) in workloads {
        for (cname, config) in configs {
            let (mean, max) = eval(values, config);
            println!("{wname},{cname},{mean:.6},{max:.6}");
        }
    }

    // Order-robustness of the global count-only update: same multiset, random
    // permutations. Recent-view knobs intentionally do not change this result.
    let mut state = 0x1234_5678u32;
    let mut worst_mean = 0.0f64;
    let mut worst_max = 0.0f64;
    for _ in 0..8 {
        let mut permuted = ascending.clone();
        for i in (1..permuted.len()).rev() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let j = (state as usize) % (i + 1);
            permuted.swap(i, j);
        }
        let (mean, max) = eval(&permuted, none);
        worst_mean = worst_mean.max(mean);
        worst_max = worst_max.max(max);
    }
    println!("permutation-worst-of-8,all-off,{worst_mean:.6},{worst_max:.6}");

    // Blocked two-mode under the neutral update — identical generator to
    // examples/adversarial_compare.rs so the numbers are comparable.
    let run_len = BATCH_LEN;
    let samples_per_mode = N / 2;
    let two_mode: Vec<f32> = (0..N)
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
        .collect();
    let (mean, max) = eval(&two_mode, none);
    println!("blocked-two-mode,all-off,{mean:.6},{max:.6}");

    // Genuine change-point latency with the default (adaptive) config:
    // uniform on [0,1], then an abrupt shift to [5,6]. How many post-shift
    // batches until the median estimate is within 0.05 of the new median?
    let mut spine = TensorDigest::<_, QuantileSpine>::with_config(&[1], default);
    let mut state = 0x9e37_79b9u32;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        ((state as f64 + 0.5) / (u32::MAX as f64 + 1.0)) as f32
    };
    for _ in 0..N {
        let v = rand();
        spine.update(&[v]).unwrap();
    }
    let mut batches_needed = None;
    for batch in 1..=20 {
        for _ in 0..BATCH_LEN {
            let v = 5.0 + rand();
            spine.update(&[v]).unwrap();
        }
        spine.flush();
        let median = spine.cell_quantiles(0, &[0.5]).unwrap()[0];
        if batches_needed.is_none() && (median - 5.5).abs() < 0.05 {
            batches_needed = Some(batch);
        }
        if batch <= 6 || batch == 20 {
            println!("post-shift batch {batch}: recent median estimate {median:.4}");
        }
    }
    println!("batches until recent median within 0.05 of new regime: {batches_needed:?}");
}

use divan::{black_box, counter::ItemsCount};
use monatq::{QuantileSpine, TensorDigest};
use statrs::distribution::{ContinuousCDF, Normal, Uniform};

fn main() {
    divan::main();
}

fn xorshift32(state: &mut u32) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state as f64) + 0.5) / (u32::MAX as f64 + 1.0)
}

fn normal_data(len: usize) -> Vec<f32> {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut state = 0xdead_beef;
    (0..len)
        .map(|_| dist.inverse_cdf(xorshift32(&mut state)) as f32)
        .collect()
}

fn uniform_data(len: usize) -> Vec<f32> {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let mut state = 0xdead_beef;
    (0..len)
        .map(|_| dist.inverse_cdf(xorshift32(&mut state)) as f32)
        .collect()
}

const SMALL_NUMEL: usize = 64 * 64;
const SMALL_SAMPLES: usize = 1_000;

#[divan::bench(sample_count = 10)]
fn tdigest_update_64x64_1k_normal(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL * SMALL_SAMPLES))
        .with_inputs(|| {
            (
                normal_data(SMALL_NUMEL * SMALL_SAMPLES),
                TensorDigest::new(&[64, 64], 100),
            )
        })
        .bench_values(|(data, mut digest)| {
            for sample in data.chunks_exact(SMALL_NUMEL) {
                digest.update(sample);
            }
            digest.flush();
            black_box((data, digest))
        });
}

#[divan::bench(sample_count = 10)]
fn spine_update_64x64_1k_normal(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL * SMALL_SAMPLES))
        .with_inputs(|| {
            (
                normal_data(SMALL_NUMEL * SMALL_SAMPLES),
                QuantileSpine::new(&[64, 64]),
            )
        })
        .bench_values(|(data, mut spine)| {
            for sample in data.chunks_exact(SMALL_NUMEL) {
                spine.update(sample);
            }
            spine.flush();
            black_box((data, spine))
        });
}

#[divan::bench(sample_count = 10)]
fn tdigest_update_64x64_1k_uniform(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL * SMALL_SAMPLES))
        .with_inputs(|| {
            (
                uniform_data(SMALL_NUMEL * SMALL_SAMPLES),
                TensorDigest::new(&[64, 64], 100),
            )
        })
        .bench_values(|(data, mut digest)| {
            for sample in data.chunks_exact(SMALL_NUMEL) {
                digest.update(sample);
            }
            digest.flush();
            black_box((data, digest))
        });
}

#[divan::bench(sample_count = 10)]
fn spine_update_64x64_1k_uniform(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL * SMALL_SAMPLES))
        .with_inputs(|| {
            (
                uniform_data(SMALL_NUMEL * SMALL_SAMPLES),
                QuantileSpine::new(&[64, 64]),
            )
        })
        .bench_values(|(data, mut spine)| {
            for sample in data.chunks_exact(SMALL_NUMEL) {
                spine.update(sample);
            }
            spine.flush();
            black_box((data, spine))
        });
}

const LARGE_NUMEL: usize = 256 * 256;
const LARGE_SAMPLES: usize = 200;

#[divan::bench(sample_count = 5)]
fn tdigest_update_256x256_200_uniform(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(LARGE_NUMEL * LARGE_SAMPLES))
        .with_inputs(|| {
            (
                uniform_data(LARGE_NUMEL * LARGE_SAMPLES),
                TensorDigest::new(&[256, 256], 100),
            )
        })
        .bench_values(|(data, mut digest)| {
            for sample in data.chunks_exact(LARGE_NUMEL) {
                digest.update(sample);
            }
            digest.flush();
            black_box((data, digest))
        });
}

#[divan::bench(sample_count = 5)]
fn spine_update_256x256_200_uniform(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(LARGE_NUMEL * LARGE_SAMPLES))
        .with_inputs(|| {
            (
                uniform_data(LARGE_NUMEL * LARGE_SAMPLES),
                QuantileSpine::new(&[256, 256]),
            )
        })
        .bench_values(|(data, mut spine)| {
            for sample in data.chunks_exact(LARGE_NUMEL) {
                spine.update(sample);
            }
            spine.flush();
            black_box((data, spine))
        });
}

#[divan::bench(sample_count = 20)]
fn tdigest_query_64x64_p99(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL))
        .with_inputs(|| {
            let data = uniform_data(SMALL_NUMEL * SMALL_SAMPLES);
            let mut digest = TensorDigest::new(&[64, 64], 100);
            for sample in data.chunks_exact(SMALL_NUMEL) {
                digest.update(sample);
            }
            digest.flush();
            digest
        })
        .bench_values(|mut digest| {
            let result = digest.quantile(0.99);
            black_box((digest, result))
        });
}

#[divan::bench(sample_count = 20)]
fn spine_query_64x64_p99(bencher: divan::Bencher) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL))
        .with_inputs(|| {
            let data = uniform_data(SMALL_NUMEL * SMALL_SAMPLES);
            let mut spine = QuantileSpine::new(&[64, 64]);
            for sample in data.chunks_exact(SMALL_NUMEL) {
                spine.update(sample);
            }
            spine.flush();
            spine
        })
        .bench_values(|mut spine| {
            let result = spine.quantile(0.99);
            black_box((spine, result))
        });
}

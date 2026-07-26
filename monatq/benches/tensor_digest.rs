use divan::{black_box, counter::ItemsCount};
use monatq::dev_support::{BACKENDS, Backend};
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
const LARGE_SAMPLES: usize = 200;

fn bench_update(
    bencher: divan::Bencher,
    backend: Backend,
    shape: &'static [usize],
    samples: usize,
    make_data: fn(usize) -> Vec<f32>,
) {
    let numel = shape.iter().product::<usize>();
    bencher
        .counter(ItemsCount::new(numel * samples))
        .with_inputs(|| (make_data(numel * samples), backend.create(shape)))
        .bench_values(|(data, mut digest)| {
            for sample in data.chunks_exact(numel) {
                digest.update(sample).unwrap();
            }
            digest.flush();
            black_box((data, digest))
        });
}

#[divan::bench(args = BACKENDS, sample_count = 10)]
fn update_64x64_1k_normal(bencher: divan::Bencher, backend: Backend) {
    bench_update(bencher, backend, &[64, 64], SMALL_SAMPLES, normal_data);
}

#[divan::bench(args = BACKENDS, sample_count = 10)]
fn update_64x64_1k_uniform(bencher: divan::Bencher, backend: Backend) {
    bench_update(bencher, backend, &[64, 64], SMALL_SAMPLES, uniform_data);
}

#[divan::bench(args = BACKENDS, sample_count = 5)]
fn update_256x256_200_uniform(bencher: divan::Bencher, backend: Backend) {
    bench_update(bencher, backend, &[256, 256], LARGE_SAMPLES, uniform_data);
}

#[divan::bench(args = BACKENDS, sample_count = 20)]
fn query_64x64_p99(bencher: divan::Bencher, backend: Backend) {
    bencher
        .counter(ItemsCount::new(SMALL_NUMEL))
        .with_inputs(|| {
            let data = uniform_data(SMALL_NUMEL * SMALL_SAMPLES);
            let mut digest = backend.create(&[64, 64]);
            for sample in data.chunks_exact(SMALL_NUMEL) {
                digest.update(sample).unwrap();
            }
            digest.flush();
            digest
        })
        .bench_values(|mut digest| {
            let result = digest.quantile(0.99);
            black_box((digest, result))
        });
}

use divan::counter::ItemsCount;
use monatq::TensorDigest;
use statrs::distribution::{ContinuousCDF, Normal, Uniform};

fn main() {
    divan::main();
}

fn xorshift32(s: &mut u32) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    (*s as f64) / (u32::MAX as f64 + 1.0)
}

#[divan::bench(sample_count = 1000)]
fn update_1k_normal(bencher: divan::Bencher) {
    const NUMEL: usize = 64 * 64;
    const SAMPLES: usize = 1000;
    let dist = Normal::new(0.0, 1.0).unwrap();

    bencher
        .counter(ItemsCount::new(NUMEL * SAMPLES))
        .with_inputs(|| {
            let mut rng = 0xDEAD_BEEFu32;
            let data = (0..SAMPLES * NUMEL)
                .map(|_| dist.inverse_cdf(xorshift32(&mut rng)) as f32)
                .collect::<Vec<f32>>();
            (data, TensorDigest::new(&[NUMEL], 100))
        })
        .bench_values(|(data, mut td)| {
            for sample in data.chunks_exact(NUMEL) {
                td.update(sample);
            }
        });
}

#[divan::bench(sample_count = 1000)]
fn update_1k_uniform(bencher: divan::Bencher) {
    const NUMEL: usize = 64 * 64;
    const SAMPLES: usize = 1000;
    let dist = Uniform::new(0.0, 1.0).unwrap();

    bencher
        .counter(ItemsCount::new(NUMEL * SAMPLES))
        .with_inputs(|| {
            let mut rng = 0xDEAD_BEEFu32;
            let data = (0..SAMPLES * NUMEL)
                .map(|_| dist.inverse_cdf(xorshift32(&mut rng)) as f32)
                .collect::<Vec<f32>>();
            (data, TensorDigest::new(&[NUMEL], 100))
        })
        .bench_values(|(data, mut td)| {
            for sample in data.chunks_exact(NUMEL) {
                td.update(sample);
            }
        });
}

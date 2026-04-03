use divan::counter::BytesCount;
use monatq::TensorDigest;
use statrs::distribution::{ContinuousCDF, Normal};

fn main() {
    divan::main();
}

const NUMEL: usize = 512 * 512;
const SAMPLES: usize = 1000;

fn xorshift32(s: &mut u32) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    (*s as f64) / (u32::MAX as f64 + 1.0)
}

fn make_digest() -> TensorDigest<f32> {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut rng = 0xDEAD_BEEFu32;
    let mut td = TensorDigest::new(&[NUMEL], 100);
    let data: Vec<f32> = (0..SAMPLES * NUMEL)
        .map(|_| dist.inverse_cdf(xorshift32(&mut rng)) as f32)
        .collect();
    for sample in data.chunks_exact(NUMEL) {
        td.update(sample);
    }
    td
}

fn saved_file_size() -> usize {
    let path = std::env::temp_dir().join("monatq_bench_size_probe.bin");
    make_digest().save(&path).unwrap();
    let size = std::fs::metadata(&path).unwrap().len() as usize;
    std::fs::remove_file(&path).ok();
    size
}

#[divan::bench]
fn save_1k_elements(bencher: divan::Bencher) {
    let path = std::env::temp_dir().join("monatq_bench_save.bin");
    let file_size = saved_file_size();

    bencher
        .counter(BytesCount::new(file_size))
        .with_inputs(make_digest)
        .bench_values(|mut td| {
            td.save(&path).unwrap();
        });
    std::fs::remove_file(&path).ok();
}

#[divan::bench]
fn load_1k_elements(bencher: divan::Bencher) {
    let path = std::env::temp_dir().join("monatq_bench_load.bin");
    make_digest().save(&path).unwrap();
    let file_size = std::fs::metadata(&path).unwrap().len() as usize;

    bencher
        .counter(BytesCount::new(file_size))
        .bench(|| TensorDigest::<f32>::load(&path).unwrap());
    std::fs::remove_file(&path).ok();
}

#[divan::bench(sample_count = 1)]
fn file_size_bytes(bencher: divan::Bencher) {
    let path = std::env::temp_dir().join("monatq_bench_size.bin");
    make_digest().save(&path).unwrap();

    bencher.bench(|| TensorDigest::<f32>::load(&path).unwrap());

    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let size = std::fs::metadata(&path).unwrap().len();
        println!("{size} bytes ({:.2} MB)", size as f64 / (1024.0 * 1024.0));
    });
    std::fs::remove_file(&path).ok();
}

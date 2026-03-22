/// cargo run --example visualize -- --help
use clap::Parser;
use monatq::TensorDigest;

// Distribution index constants matching Distribution::iter() order.
const KIND_NORMAL: usize = 0;
const KIND_UNIFORM: usize = 1;
const KIND_LAPLACE: usize = 2;
const KIND_LOGNORMAL: usize = 3;

/// Feed synthetic data into a TensorDigest and open the browser visualizer.
#[derive(Parser)]
struct Args {
    /// Tensor shape as comma-separated dimensions, e.g. 2,3,12,16.
    /// Must have at least 2 dimensions (H,W). Leading dims become batch/channel axes.
    #[arg(default_value = "2,3,12,16")]
    shape: Shape,

    /// Number of samples to feed into the digest.
    #[arg(short, long, default_value_t = 2_000)]
    samples: u32,

    /// Compression parameter for each kernel.
    #[arg(short, long, default_value_t = 100)]
    compression: usize,
}

#[derive(Clone)]
struct Shape(Vec<usize>);

impl std::str::FromStr for Shape {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_shape(s).map(Shape)
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let shape = &args.shape.0;
    let numel: usize = shape.iter().product();
    let mut td = TensorDigest::new(shape, args.compression);

    // Assign a random kind to every element position.
    // Kinds 0..3 are Normal/Uniform/Laplace/LogNormal; kind 4 is bimodal (Unknown).
    let n_kinds = 5;

    let mut rng = Rng::new(0xdeadbeef);

    let element_kinds: Vec<usize> = (0..numel)
        .map(|_| rng.next_u32() as usize % n_kinds)
        .collect();

    let n_samples = args.samples;
    for _ in 0..n_samples {
        let mut frame = vec![0f32; numel];
        for (v, &element_kind) in frame.iter_mut().zip(element_kinds.iter()) {
            *v = match element_kind {
                KIND_NORMAL => rng.normal(),
                KIND_UNIFORM => rng.uniform_pm_sqrt3(),
                KIND_LAPLACE => rng.laplace(),
                KIND_LOGNORMAL => rng.lognormal_standardized(),
                // bimodal: should be classified as Unknown
                _ => {
                    if rng.f32() < 0.5 {
                        rng.normal() - 3.0
                    } else {
                        rng.normal() + 3.0
                    }
                }
            };
        }
        td.update(&frame);
    }

    eprintln!("fed {n_samples} samples  shape {shape:?}");
    td.visualize()
}

fn parse_shape(s: &str) -> Result<Vec<usize>, String> {
    let dims: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    if dims.len() < 2 {
        return Err("shape must have at least 2 dimensions".into());
    }
    if dims.iter().any(|&d| d == 0) {
        return Err("shape dimensions must be non-zero".into());
    }
    Ok(dims)
}

// ── minimal PRNG (xorshift32) ────────────────────────────────────────────────

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform [0, 1)
    fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal via Box-Muller
    fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Uniform on [-√3, √3]: mean=0, variance=1
    fn uniform_pm_sqrt3(&mut self) -> f32 {
        let sqrt3 = 3f32.sqrt();
        self.f32() * 2.0 * sqrt3 - sqrt3
    }

    /// Laplace(0, 1/√2): mean=0, variance=1, via inverse-CDF
    fn laplace(&mut self) -> f32 {
        let u = self.f32() - 0.5;
        let scale = std::f32::consts::FRAC_1_SQRT_2;
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// LogNormal(0,1) shifted/scaled to mean=0, variance=1
    fn lognormal_standardized(&mut self) -> f32 {
        const MEAN: f32 = 1.648_721_3;
        const STD: f32 = 2.1612;
        (self.normal().exp() - MEAN) / STD
    }
}

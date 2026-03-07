pub struct TDigest {
    // SoA layout — enables direct f32x8 loads
    means: Vec<f32>,
    weights: Vec<f32>,

    // Unmerged input buffer — sorted and flushed lazily
    buffer: Vec<f32>,

    total_weight: f32,
    compression: f32,

    // Capacity hint to avoid reallocs
    max_centroids: usize,
}

impl TDigest {
    pub fn new(compression: f32) -> Self {
        // Theoretical max from the paper: π * compression / 2
        // Use * 2 for safety without being wasteful
        let max_centroids = (compression * 2.0) as usize;
        let buffer_capacity = (compression * 5.0) as usize;

        Self {
            means: Vec::with_capacity(max_centroids),
            weights: Vec::with_capacity(max_centroids),
            buffer: Vec::with_capacity(buffer_capacity),
            total_weight: 0.0,
            compression,
            max_centroids,
        }
    }

    #[inline(always)]
    fn k2_limit(&self, q: f32) -> f32 {
        self.compression * q * (1.0 - q)
    }

    /// Weighted mean of two centroids
    #[inline(always)]
    fn weighted_mean(m1: f32, w1: f32, m2: f32, w2: f32) -> f32 {
        (m1 * w1 + m2 * w2) / (w1 + w2)
    }

    pub fn quantile(&mut self, q: f32) -> f32 {
        if !self.buffer.is_empty() {
            self.compress();
        }
        let target = q * self.total_weight;
        let mut cumulative = 0.0f32;

        for ((&mean, &weight), next_mean) in self.means.iter().zip(self.weights.iter()).zip(
            self.means
                .iter()
                .skip(1)
                .map(|&m| Some(m))
                .chain(std::iter::once(None)),
        ) {
            cumulative += weight;
            if cumulative >= target {
                if let Some(nm) = next_mean {
                    // Linear interpolation between centroids
                    let t = (target - (cumulative - weight)) / weight;
                    return mean + t * (nm - mean);
                }
                return mean;
            }
        }

        *self.means.last().unwrap_or(&0.0)
    }

    fn compress(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        // Sort buffer — in-place, no scratch allocation
        self.buffer.sort_unstable_by(f32::total_cmp);

        // Merge buffer into centroids
        let mut new_means: Vec<f32> = Vec::with_capacity(self.max_centroids);
        let mut new_weights: Vec<f32> = Vec::with_capacity(self.max_centroids);

        // Combine existing centroids + buffer into a single sorted pass
        let mut ci = 0usize; // centroid index
        let mut bi = 0usize; // buffer index

        let mut cur_mean = 0.0f32;
        let mut cur_weight = 0.0f32;
        let mut cumulative = 0.0f32;
        let total = self.total_weight + self.buffer.len() as f32;

        macro_rules! next_point {
            () => {{
                // Merge-sort step: pick smaller of centroid or buffer head
                let from_centroid = ci < self.means.len();
                let from_buffer = bi < self.buffer.len();
                match (from_centroid, from_buffer) {
                    (true, true) if self.means[ci] <= self.buffer[bi] => {
                        let m = self.means[ci];
                        let w = self.weights[ci];
                        ci += 1;
                        (m, w)
                    }
                    (_, true) => {
                        let m = self.buffer[bi];
                        bi += 1;
                        (m, 1.0f32)
                    }
                    (true, false) => {
                        let m = self.means[ci];
                        let w = self.weights[ci];
                        ci += 1;
                        (m, w)
                    }
                    _ => unreachable!(),
                }
            }};
        }

        let n = self.means.len() + self.buffer.len();
        let mut remaining = n;

        while remaining > 0 {
            let (m, w) = next_point!();
            remaining -= 1;

            let q = cumulative / total;
            let limit = self.k2_limit(q);

            if cur_weight + w <= limit {
                // Merge into current centroid — SIMD-able in bulk version
                cur_mean = Self::weighted_mean(cur_mean, cur_weight, m, w);
                cur_weight += w;
            } else {
                // Emit current, start new
                if cur_weight > 0.0 {
                    new_means.push(cur_mean);
                    new_weights.push(cur_weight);
                }
                cumulative += cur_weight;
                cur_mean = m;
                cur_weight = w;
            }
        }

        // Flush last centroid
        if cur_weight > 0.0 {
            new_means.push(cur_mean);
            new_weights.push(cur_weight);
        }

        self.means = new_means;
        self.weights = new_weights;
        self.total_weight = total;
        self.buffer.clear();
    }

    #[inline(always)]
    fn max_buffer_len(&self) -> usize {
        (self.compression * 5.0) as usize
    }

    #[inline]
    pub fn add(&mut self, value: f32) {
        self.buffer.push(value);
        if self.buffer.len() >= self.max_buffer_len() {
            self.compress();
        }
    }
}

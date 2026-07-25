use rayon::prelude::*;
use std::{
    mem::size_of,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};
use wide::f32x8;

use crate::{
    TensorValue,
    kernels::{DigestKernel, QuantileSpine, sealed},
    tensor_digest::StorageOperations,
};

/// Number of quantile anchors in each spine.
pub const SPINE_K: usize = 64;
/// Number of exact all-time records retained at each end.
pub const RECORDS_T: usize = 8;
/// Number of exact records retained at each end since the last restart.
pub const WINDOW_RECORDS_T: usize = 4;

const PROBIT_EPSILON: f32 = 1.0e-6;
const LINK_SWITCH_MARGIN: f32 = 0.2;
const PROBIT_LINK: u8 = 0;
const LINEAR_LINK: u8 = 1;
const LOG_PROBIT_LINK: u8 = 2;
const REFIT_SAMPLE_POSITIONS: usize = 256;
const REFIT_EDGE_ANCHORS: usize = 4;
const GAUGE_MASK: u32 = 0xffff;
const REGIME_SHIFT: u32 = 16;
const REGIME_MASK: u32 = 0b11 << REGIME_SHIFT;
const CROSSINGS_SHIFT: u32 = 18;
const CROSSINGS_MASK: u32 = 0b11 << CROSSINGS_SHIFT;
const SUPPRESSION_SHIFT: u32 = 20;
const SUPPRESSION_MASK: u32 = 0b111 << SUPPRESSION_SHIFT;
const WINDOW_COUNT_SHIFT: u32 = 23;
const WINDOW_COUNT_MASK: u32 = 0b111 << WINDOW_COUNT_SHIFT;
const SECONDARY_CANDIDATE_MASK: u32 = 1 << 26;

/// Adaptation state encoded in a position's surprise word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpineRegime {
    Calm,
    Alert,
    Restart,
}

/// Shared interpolation ruler used by every position in a tensor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpineLink {
    /// Interpolate values against inverse-normal transformed ranks.
    #[default]
    Probit,
    /// Interpolate values directly against ranks.
    Linear,
    /// Interpolate log-values against inverse-normal transformed ranks.
    LogProbit,
}

impl SpineLink {
    fn rank_coordinate(self, rank: f32) -> f32 {
        match self {
            Self::Linear => rank,
            Self::Probit | Self::LogProbit => probit(rank),
        }
    }

    fn rank_from_coordinate(self, coordinate: f32) -> f32 {
        match self {
            Self::Linear => coordinate.clamp(0.0, 1.0),
            Self::Probit | Self::LogProbit => normal_cdf(coordinate),
        }
    }

    fn transformed_grid(self) -> &'static [f32; SPINE_K] {
        match self {
            Self::Linear => rank_grid(),
            Self::Probit | Self::LogProbit => probit_grid(),
        }
    }

    fn value_coordinate(self, value: f32) -> f32 {
        match self {
            Self::LogProbit if value > 0.0 => value.ln(),
            Self::LogProbit => f32::NEG_INFINITY,
            Self::Probit | Self::Linear => value,
        }
    }

    fn value_from_coordinate(self, coordinate: f32) -> f32 {
        match self {
            Self::LogProbit => coordinate.exp(),
            Self::Probit | Self::Linear => coordinate,
        }
    }

    const fn score_index(self) -> usize {
        match self {
            Self::Probit => 0,
            Self::Linear => 1,
            Self::LogProbit => 2,
        }
    }
}

/// Shared configuration for the [`crate::QuantileSpine`] kernel.
#[derive(Clone, Copy, Debug)]
pub struct QuantileSpineConfig {
    /// Number of tensor samples collected before a parallel flush.
    pub buffer_capacity: usize,
    /// Maximum history weight used by blending. `u64::MAX` disables fading.
    pub n_max: u64,
    /// DKW false-alert probability used to derive the KS threshold.
    pub beta: f32,
    /// Surprise-gain coefficient `c` in `exp(-c B (D - tau)^2)`.
    pub gain_c: f32,
    /// EWMA gain for the quantized surprise gauge.
    pub gauge_rho: f32,
    /// Minimum tied fraction required to promote a nonzero secondary atom.
    pub atom_threshold: f32,
    /// Consecutive threshold crossings needed for a restart. Zero disables restarts.
    pub restart_crossings: u8,
    /// Number of flushes (including the triggering flush) whose history is suppressed.
    pub restart_batches: u8,
    /// Flushes between shared-ruler link refits. Zero keeps the probit link fixed.
    pub link_refit_interval: usize,
    /// Seed for the shared rank-grid dither.
    pub dither_seed: u64,
}

impl Default for QuantileSpineConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 256,
            n_max: 100_000,
            beta: 0.01,
            gain_c: 2.0,
            gauge_rho: 0.1,
            atom_threshold: 0.1,
            restart_crossings: 3,
            restart_batches: 3,
            link_refit_interval: 16,
            dither_seed: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PositionMeta {
    zero_count: u32,
    secondary_value: f32,
    secondary_count: u32,
    surprise_word: u32,
}

const _: () = assert!(size_of::<PositionMeta>() == 16);

/// A flat-array Quantile Spine for row-major tensor streams.
///
/// Every position owns 64 `f32` anchors, exact record shelves, two atom counters,
/// and one packed surprise word. All arrays are contiguous; no position performs a
/// heap allocation. Incoming rows use the same `row_buffer[s * numel + i]` layout
/// as [`crate::TensorDigest`], and flushes are parallel over positions.
pub struct QuantileSpineStorage<T: TensorValue> {
    shape: Vec<usize>,
    numel: usize,
    config: QuantileSpineConfig,

    row_buffer: Vec<T>,
    sort_buffer: Vec<f32>,
    n_buffered: usize,
    total_count: u32,

    anchors: Vec<f32>,
    low_records: Vec<f32>,
    high_records: Vec<f32>,
    window_low_records: Vec<f32>,
    window_high_records: Vec<f32>,
    metadata: Vec<PositionMeta>,

    link: SpineLink,
    flush_count: u64,
    dither_state: u64,
}

/// Alternate name emphasizing the tensor-oriented storage layout.
impl<T: TensorValue> QuantileSpineStorage<T> {
    /// Construct a spine with explicit shared configuration.
    pub fn with_config(shape: &[usize], config: QuantileSpineConfig) -> Self {
        assert!(
            config.buffer_capacity > 0,
            "buffer_capacity must be positive"
        );
        assert!(
            config.beta > 0.0 && config.beta < 1.0,
            "beta must be in (0, 1)"
        );
        assert!(
            config.gauge_rho > 0.0 && config.gauge_rho <= 1.0,
            "gauge_rho must be in (0, 1]"
        );
        assert!(
            config.atom_threshold > 0.0 && config.atom_threshold <= 1.0,
            "atom_threshold must be in (0, 1]"
        );
        assert!(
            config.restart_crossings <= 3,
            "the packed surprise word supports at most three crossings"
        );
        assert!(
            config.restart_batches <= 7,
            "the packed surprise word supports at most seven suppressed batches"
        );

        let numel = shape.iter().product::<usize>();
        let row_len = numel
            .checked_mul(config.buffer_capacity)
            .expect("row buffer size overflow");
        Self {
            shape: shape.to_vec(),
            numel,
            config,
            row_buffer: vec![T::from_f32(0.0); row_len],
            sort_buffer: vec![0.0; row_len],
            n_buffered: 0,
            total_count: 0,
            anchors: vec![0.0; numel * SPINE_K],
            low_records: vec![0.0; numel * RECORDS_T],
            high_records: vec![0.0; numel * RECORDS_T],
            window_low_records: vec![0.0; numel * WINDOW_RECORDS_T],
            window_high_records: vec![0.0; numel * WINDOW_RECORDS_T],
            metadata: vec![PositionMeta::default(); numel],
            link: SpineLink::Probit,
            flush_count: 0,
            dither_state: if config.dither_seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                config.dither_seed
            },
        }
    }

    /// Total number of positions in each tensor sample.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Tensor shape supplied at construction.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Shared number of flushed samples. As with `TensorDigest::total_weight`,
    /// rows still in the input buffer are not included until a flush.
    pub fn total_weight(&self, idx: usize) -> u32 {
        assert!(idx < self.numel, "position index out of bounds");
        self.total_count
    }

    /// Shared number of flushed samples.
    pub fn sample_count(&self) -> u32 {
        self.total_count
    }

    /// Current shared configuration.
    pub fn config(&self) -> &QuantileSpineConfig {
        &self.config
    }

    /// Current tensor-wide interpolation link.
    pub fn link(&mut self) -> SpineLink {
        self.flush();
        self.link
    }

    /// Add one row-major tensor sample.
    pub fn update(&mut self, data: &[T]) {
        assert_eq!(
            data.len(),
            self.numel,
            "data length {} does not match numel {}",
            data.len(),
            self.numel
        );
        let start = self.n_buffered * self.numel;
        self.row_buffer[start..start + self.numel].copy_from_slice(data);
        self.n_buffered += 1;
        if self.n_buffered == self.config.buffer_capacity {
            self.flush();
        }
    }

    /// Fold all buffered rows into the per-position spines.
    pub fn flush(&mut self) {
        if self.n_buffered == 0 {
            return;
        }

        let batch_len = self.n_buffered;
        let old_n = self.total_count as usize;
        let new_count = self
            .total_count
            .checked_add(batch_len.try_into().expect("batch length exceeds u32"))
            .expect("QuantileSpine sample count exceeds u32::MAX");
        let new_n = new_count as usize;
        let numel = self.numel;
        let config = self.config;
        let tau = (f32::ln(2.0 / config.beta) / (2.0 * batch_len as f32)).sqrt();
        let eta = self.next_dither();
        let targets = shifted_ranks(eta);
        let link = self.link;
        let target_axis = targets.map(|rank| link.rank_coordinate(rank));
        let batch_interpolation = sample_interpolation(&targets, batch_len, link);
        let row_buffer = &self.row_buffer;
        let sorted_batches = &mut self.sort_buffer[..numel * batch_len];
        sort_batches_simd(row_buffer, numel, batch_len, sorted_batches);

        let anchors = &mut self.anchors;
        let low_records = &mut self.low_records;
        let high_records = &mut self.high_records;
        let window_low_records = &mut self.window_low_records;
        let window_high_records = &mut self.window_high_records;
        let metadata = &mut self.metadata;
        let restarted_any = AtomicBool::new(false);

        anchors
            .par_chunks_mut(SPINE_K)
            .zip(low_records.par_chunks_mut(RECORDS_T))
            .zip(high_records.par_chunks_mut(RECORDS_T))
            .zip(window_low_records.par_chunks_mut(WINDOW_RECORDS_T))
            .zip(window_high_records.par_chunks_mut(WINDOW_RECORDS_T))
            .zip(metadata.par_iter_mut())
            .enumerate()
            .for_each_init(
                || FlushScratch::new(batch_len),
                |scratch, (e, (((((anchors, low), high), window_low), window_high), meta))| {
                    scratch.batch.clear();
                    scratch
                        .batch
                        .extend_from_slice(&sorted_batches[e * batch_len..(e + 1) * batch_len]);
                    scratch.old.copy_from_slice(anchors);
                    let old_meta = *meta;

                    let surprise = if old_n == 0 {
                        0.0
                    } else if old_n <= SPINE_K {
                        ks_distance_exact(&anchors[..old_n], &scratch.batch, old_n as f32)
                    } else {
                        ks_distance_screened(anchors, &scratch.batch, old_n, old_meta, tau, link)
                    };
                    let (restarted, suppress_history) =
                        update_surprise(&mut meta.surprise_word, surprise, tau, config);
                    if restarted {
                        restarted_any.store(true, Ordering::Relaxed);
                    }

                    let old_record_len = old_n.min(RECORDS_T);
                    merge_low::<RECORDS_T>(
                        &low[..old_record_len],
                        &scratch.batch,
                        &mut scratch.low_records,
                    );
                    merge_high::<RECORDS_T>(
                        &high[..old_record_len],
                        &scratch.batch,
                        &mut scratch.high_records,
                    );
                    low.copy_from_slice(&scratch.low_records);
                    high.copy_from_slice(&scratch.high_records);

                    let old_window_len = if restarted {
                        0
                    } else {
                        window_count(meta.surprise_word).min(WINDOW_RECORDS_T)
                    };
                    merge_low::<WINDOW_RECORDS_T>(
                        &window_low[..old_window_len],
                        &scratch.batch,
                        &mut scratch.window_low_records,
                    );
                    merge_high::<WINDOW_RECORDS_T>(
                        &window_high[..old_window_len],
                        &scratch.batch,
                        &mut scratch.window_high_records,
                    );
                    window_low.copy_from_slice(&scratch.window_low_records);
                    window_high.copy_from_slice(&scratch.window_high_records);
                    set_window_count(
                        &mut meta.surprise_word,
                        (old_window_len + batch_len).min(WINDOW_RECORDS_T),
                    );

                    let batch_zeros = scratch.batch.iter().filter(|&&x| x == 0.0).count();
                    meta.zero_count = meta
                        .zero_count
                        .checked_add(batch_zeros as u32)
                        .expect("zero counter exceeds u32::MAX");

                    if old_n <= SPINE_K {
                        merge_sorted(&anchors[..old_n], &scratch.batch, &mut scratch.merged);
                        if new_n <= SPINE_K {
                            anchors[..new_n].copy_from_slice(&scratch.merged);
                            anchors[new_n..].fill(0.0);
                            // Early life is represented by the exact sorted state; atom
                            // promotion starts when the stream first outgrows that state.
                            meta.secondary_count = 0;
                            set_secondary_candidate(&mut meta.surprise_word, false);
                            return;
                        }

                        if let Some((value, _)) = find_modal_tie(
                            &scratch.merged,
                            config.atom_threshold,
                            scratch.merged.len(),
                        ) {
                            meta.secondary_value = value;
                            meta.secondary_count =
                                scratch.merged.iter().filter(|&&x| x == value).count() as u32;
                            set_secondary_candidate(&mut meta.surprise_word, false);
                        } else {
                            meta.secondary_count = 0;
                            set_secondary_candidate(&mut meta.surprise_word, false);
                        }

                        scratch.smooth.clear();
                        scratch
                            .smooth
                            .extend(scratch.merged.iter().copied().filter(|&x| {
                                x != 0.0 && (meta.secondary_count == 0 || x != meta.secondary_value)
                            }));
                        // There is no interpolation error to randomize when the exact
                        // early state first becomes a spine, so initialize canonically.
                        empirical_reanchor(&scratch.smooth, rank_grid(), &mut scratch.output);
                        anchors.copy_from_slice(&scratch.output);
                        return;
                    }

                    let old_secondary_active = old_meta.secondary_count > 0
                        && !secondary_candidate(old_meta.surprise_word);
                    let old_secondary_count = if old_secondary_active {
                        old_meta.secondary_count as usize
                    } else {
                        0
                    };
                    if old_secondary_active {
                        let added = scratch
                            .batch
                            .iter()
                            .filter(|&&x| x == old_meta.secondary_value)
                            .count();
                        meta.secondary_count = meta
                            .secondary_count
                            .checked_add(added as u32)
                            .expect("secondary atom counter exceeds u32::MAX");
                        set_secondary_candidate(&mut meta.surprise_word, false);
                    } else {
                        match find_modal_tie(&scratch.batch, config.atom_threshold, batch_len) {
                            Some((value, count))
                                if secondary_candidate(old_meta.surprise_word)
                                    && value == old_meta.secondary_value =>
                            {
                                // A second consecutive tied batch confirms promotion.
                                // Counting starts at this promotion boundary; the candidate
                                // batch remains part of the smooth historical spine.
                                meta.secondary_value = value;
                                meta.secondary_count = count as u32;
                                set_secondary_candidate(&mut meta.surprise_word, false);
                            }
                            Some((value, count)) => {
                                meta.secondary_value = value;
                                meta.secondary_count = count as u32;
                                set_secondary_candidate(&mut meta.surprise_word, true);
                            }
                            None => {
                                meta.secondary_count = 0;
                                set_secondary_candidate(&mut meta.surprise_word, false);
                            }
                        }
                    }

                    let secondary_active =
                        meta.secondary_count > 0 && !secondary_candidate(meta.surprise_word);
                    scratch.smooth.clear();
                    scratch.smooth.extend(
                        scratch.batch.iter().copied().filter(|&x| {
                            x != 0.0 && (!secondary_active || x != meta.secondary_value)
                        }),
                    );

                    let old_smooth_count = old_n
                        .saturating_sub(old_meta.zero_count as usize)
                        .saturating_sub(old_secondary_count);
                    if scratch.smooth.is_empty() {
                        // Atom-only batches change their exact masses but contain no
                        // information with which to re-read the conditional smooth part.
                        return;
                    }
                    if old_smooth_count == 0 {
                        empirical_reanchor(&scratch.smooth, rank_grid(), &mut scratch.output);
                        anchors.copy_from_slice(&scratch.output);
                        return;
                    }

                    let n_eff = (old_n as u64).min(config.n_max) as f32;
                    let base_new_gain = batch_len as f32 / (n_eff + batch_len as f32);
                    // A DKW crossing at tau is expected with probability beta. Letting one
                    // such batch discard most history created rare ~3% rank outliers on
                    // stationary streams. Gate the gain at 1.5 tau, but retain the paper's
                    // original (D - tau) magnitude once the evidence is strong; abrupt
                    // shifts therefore keep their original one-batch response.
                    let excess = if surprise > 1.5 * tau {
                        surprise - tau
                    } else {
                        0.0
                    };
                    let surprise_exponent = config.gain_c * batch_len as f32 * excess * excess;
                    let surprise_new_gain = -f32::exp_m1(-surprise_exponent);
                    let new_gain = if suppress_history {
                        1.0
                    } else {
                        base_new_gain + surprise_new_gain * (1.0 - base_new_gain)
                    };
                    // Below the DKW threshold the two smooth distributions are
                    // statistically indistinguishable. Their CDF-mixture quantiles agree
                    // to second order with this quantile-space (Wasserstein) blend, which
                    // is branch-free O(K). Surprising or atomic batches retain the exact
                    // coordinated CDF sweep below.
                    if surprise < tau
                        && old_meta.zero_count == 0
                        && old_meta.secondary_count == 0
                        && batch_zeros == 0
                    {
                        blend_reanchor_close(
                            &scratch.old,
                            &scratch.smooth,
                            new_gain,
                            &targets,
                            &batch_interpolation,
                            link,
                            &mut scratch.output,
                        );
                    } else {
                        blend_reanchor(
                            &scratch.old,
                            &scratch.smooth,
                            new_gain,
                            &targets,
                            &target_axis,
                            link,
                            &mut scratch.output,
                        );
                    }
                    anchors.copy_from_slice(&scratch.output);
                },
            );

        self.total_count = new_count;
        self.n_buffered = 0;
        self.flush_count = self.flush_count.wrapping_add(1);

        let refits_enabled = config.link_refit_interval != 0;
        let left_early_life = old_n <= SPINE_K && new_n > SPINE_K;
        let periodic_refit =
            refits_enabled && self.flush_count % config.link_refit_interval as u64 == 0;
        let log_link_invalid = self.link == SpineLink::LogProbit
            && !log_link_is_valid(&self.anchors, &self.low_records, self.numel);
        if refits_enabled
            && (left_early_life
                || periodic_refit
                || restarted_any.load(Ordering::Relaxed)
                || log_link_invalid)
        {
            self.refit_link();
        }
    }

    /// Compute one quantile at every tensor position.
    pub fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.flush();
        self.quantile_no_flush(q)
    }

    /// Compute several quantiles at every tensor position.
    pub fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.flush();
        qs.iter().map(|&q| self.quantile_no_flush(q)).collect()
    }

    /// Compute several quantiles for one flat-indexed tensor position.
    pub fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        let anchor_start = idx * SPINE_K;
        let record_start = idx * RECORDS_T;
        let anchors = &self.anchors[anchor_start..anchor_start + SPINE_K];
        let low = &self.low_records[record_start..record_start + RECORDS_T];
        let high = &self.high_records[record_start..record_start + RECORDS_T];
        let meta = self.metadata[idx];
        match self.link {
            SpineLink::Probit => quantiles_for_cell::<PROBIT_LINK>(
                anchors,
                low,
                high,
                meta,
                self.total_count as usize,
                qs,
            ),
            SpineLink::Linear => quantiles_for_cell::<LINEAR_LINK>(
                anchors,
                low,
                high,
                meta,
                self.total_count as usize,
                qs,
            ),
            SpineLink::LogProbit => quantiles_for_cell::<LOG_PROBIT_LINK>(
                anchors,
                low,
                high,
                meta,
                self.total_count as usize,
                qs,
            ),
        }
    }

    /// Exact all-time minima, one per position.
    pub fn min(&mut self) -> Vec<f32> {
        self.flush();
        if self.total_count == 0 {
            return vec![0.0; self.numel];
        }
        self.low_records
            .par_chunks(RECORDS_T)
            .map(|shelf| shelf[0])
            .collect()
    }

    /// Exact all-time maxima, one per position.
    pub fn max(&mut self) -> Vec<f32> {
        self.flush();
        if self.total_count == 0 {
            return vec![0.0; self.numel];
        }
        let len = (self.total_count as usize).min(RECORDS_T);
        self.high_records
            .par_chunks(RECORDS_T)
            .map(|shelf| shelf[len - 1])
            .collect()
    }

    /// Exact all-time minimum for one flat-indexed position.
    pub fn cell_min(&mut self, idx: usize) -> f32 {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        if self.total_count == 0 {
            0.0
        } else {
            self.low_records[idx * RECORDS_T]
        }
    }

    /// Exact all-time maximum for one flat-indexed position.
    pub fn cell_max(&mut self, idx: usize) -> f32 {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        if self.total_count == 0 {
            0.0
        } else {
            let len = (self.total_count as usize).min(RECORDS_T);
            self.high_records[idx * RECORDS_T + len - 1]
        }
    }

    /// Exact minima since each position's most recent restart.
    pub fn recent_min(&mut self) -> Vec<f32> {
        self.flush();
        self.window_low_records
            .par_chunks(WINDOW_RECORDS_T)
            .zip(self.metadata.par_iter())
            .map(|(shelf, meta)| {
                if window_count(meta.surprise_word) == 0 {
                    0.0
                } else {
                    shelf[0]
                }
            })
            .collect()
    }

    /// Exact maxima since each position's most recent restart.
    pub fn recent_max(&mut self) -> Vec<f32> {
        self.flush();
        self.window_high_records
            .par_chunks(WINDOW_RECORDS_T)
            .zip(self.metadata.par_iter())
            .map(|(shelf, meta)| {
                let len = window_count(meta.surprise_word);
                if len == 0 { 0.0 } else { shelf[len - 1] }
            })
            .collect()
    }

    /// Exact count of `0.0` at one position.
    pub fn zero_count(&mut self, idx: usize) -> u32 {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        self.metadata[idx].zero_count
    }

    /// Promoted nonzero atom and its exact count since promotion.
    pub fn secondary_atom(&mut self, idx: usize) -> Option<(f32, u32)> {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        let meta = self.metadata[idx];
        (meta.secondary_count > 0 && !secondary_candidate(meta.surprise_word))
            .then_some((meta.secondary_value, meta.secondary_count))
    }

    /// Quantized EWMA of recent one-sample KS surprise at one position.
    pub fn surprise(&mut self, idx: usize) -> f32 {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        gauge(self.metadata[idx].surprise_word)
    }

    /// Current regime-machine state at one position.
    pub fn regime(&mut self, idx: usize) -> SpineRegime {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        regime(self.metadata[idx].surprise_word)
    }

    fn quantile_no_flush(&self, q: f32) -> Vec<f32> {
        match self.link {
            SpineLink::Probit => self.quantile_no_flush_for::<PROBIT_LINK>(q),
            SpineLink::Linear => self.quantile_no_flush_for::<LINEAR_LINK>(q),
            SpineLink::LogProbit => self.quantile_no_flush_for::<LOG_PROBIT_LINK>(q),
        }
    }

    fn quantile_no_flush_for<const LINK: u8>(&self, q: f32) -> Vec<f32> {
        let n = self.total_count as usize;
        let interpolation = query_interpolation::<LINK>(q);
        if LINK == LINEAR_LINK && self.numel <= 8_192 {
            return self
                .anchors
                .chunks(SPINE_K)
                .zip(self.low_records.chunks(RECORDS_T))
                .zip(self.high_records.chunks(RECORDS_T))
                .zip(self.metadata.iter())
                .map(|(((anchors, low), high), &meta)| {
                    quantile_element::<LINK>(anchors, low, high, meta, n, q, interpolation)
                })
                .collect();
        }
        self.anchors
            .par_chunks(SPINE_K)
            .zip(self.low_records.par_chunks(RECORDS_T))
            .zip(self.high_records.par_chunks(RECORDS_T))
            .zip(self.metadata.par_iter())
            .map(|(((anchors, low), high), &meta)| {
                quantile_element::<LINK>(anchors, low, high, meta, n, q, interpolation)
            })
            .collect()
    }

    fn refit_link(&mut self) {
        let scores = shared_link_scores(
            &self.anchors,
            &self.low_records,
            self.numel,
            self.total_count as usize,
        );
        let candidates = [SpineLink::Probit, SpineLink::Linear, SpineLink::LogProbit];
        let winner = candidates
            .into_iter()
            .min_by(|left, right| {
                scores[left.score_index()].total_cmp(&scores[right.score_index()])
            })
            .unwrap_or(self.link);
        let winner_score = scores[winner.score_index()];
        let incumbent_score = scores[self.link.score_index()];
        if winner != self.link
            && winner_score.is_finite()
            && (!incumbent_score.is_finite()
                || winner_score < (1.0 - LINK_SWITCH_MARGIN) * incumbent_score)
        {
            self.link = winner;
        }
    }

    fn next_dither(&mut self) -> f32 {
        let mut x = self.dither_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.dither_state = x;
        let unit = (x >> 40) as f32 / (1u32 << 24) as f32;
        unit - 0.5
    }
}

impl<T: TensorValue> StorageOperations<T> for QuantileSpineStorage<T> {
    fn numel(&self) -> usize {
        self.numel()
    }
    fn shape(&self) -> &[usize] {
        self.shape()
    }
    fn total_weight(&self, idx: usize) -> u32 {
        self.total_weight(idx)
    }
    fn update(&mut self, data: &[T]) {
        self.update(data)
    }
    fn flush(&mut self) {
        self.flush()
    }
    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.quantile(q)
    }
    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.quantiles(qs)
    }
    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.cell_quantiles(idx, qs)
    }
}

impl<T: TensorValue> DigestKernel<T> for QuantileSpine {
    type Config = QuantileSpineConfig;
}

impl<T: TensorValue> sealed::Kernel<T> for QuantileSpine {
    type Storage = QuantileSpineStorage<T>;

    fn create_storage(
        shape: &[usize],
        config: <QuantileSpine as DigestKernel<T>>::Config,
    ) -> Self::Storage {
        QuantileSpineStorage::with_config(shape, config)
    }
}

fn log_link_is_valid(anchors: &[f32], low_records: &[f32], numel: usize) -> bool {
    numel > 0
        && anchors
            .chunks_exact(SPINE_K)
            .all(|curve| curve[0].is_finite() && curve[0] > 0.0 && curve[SPINE_K - 1].is_finite())
        && low_records
            .chunks_exact(RECORDS_T)
            .all(|shelf| shelf[0].is_finite() && shelf[0] > 0.0)
}

fn pooled_normalized_curve(
    anchors: &[f32],
    numel: usize,
    link: SpineLink,
) -> Option<[f64; SPINE_K]> {
    let sample_count = numel.min(REFIT_SAMPLE_POSITIONS);
    if sample_count == 0 {
        return None;
    }

    let lower = REFIT_EDGE_ANCHORS;
    let upper = SPINE_K - 1 - REFIT_EDGE_ANCHORS;
    let mut pooled = [0.0f64; SPINE_K];
    let mut curves = 0usize;
    for sample in 0..sample_count {
        let position = sample * numel / sample_count;
        let curve = &anchors[position * SPINE_K..(position + 1) * SPINE_K];
        let y0 = link.value_coordinate(curve[lower]) as f64;
        let y1 = link.value_coordinate(curve[upper]) as f64;
        let scale = y1 - y0;
        if !y0.is_finite()
            || !y1.is_finite()
            || scale <= f64::EPSILON * y0.abs().max(y1.abs()).max(1.0)
        {
            continue;
        }
        if curve.iter().any(|value| {
            let coordinate = link.value_coordinate(*value);
            !coordinate.is_finite()
        }) {
            continue;
        }
        for (sum, &value) in pooled.iter_mut().zip(curve) {
            *sum += (link.value_coordinate(value) as f64 - y0) / scale;
        }
        curves += 1;
    }
    if curves == 0 {
        return None;
    }
    for value in &mut pooled {
        *value /= curves as f64;
    }
    Some(pooled)
}

fn straightness_score(curve: &[f64; SPINE_K], axis: &[f32; SPINE_K]) -> f32 {
    let first = REFIT_EDGE_ANCHORS;
    let end = SPINE_K - REFIT_EDGE_ANCHORS;
    let points = (end - first) as f64;
    let mean_x = axis[first..end]
        .iter()
        .map(|&value| value as f64)
        .sum::<f64>()
        / points;
    let mean_y = curve[first..end].iter().sum::<f64>() / points;
    let mut covariance = 0.0f64;
    let mut variance = 0.0f64;
    for index in first..end {
        let dx = axis[index] as f64 - mean_x;
        covariance += dx * (curve[index] - mean_y);
        variance += dx * dx;
    }
    if variance <= f64::EPSILON {
        return f32::INFINITY;
    }
    let slope = covariance / variance;
    let intercept = mean_y - slope * mean_x;
    let energy = (first..end)
        .map(|index| {
            let residual = curve[index] - (intercept + slope * axis[index] as f64);
            residual * residual
        })
        .sum::<f64>()
        / points;
    energy as f32
}

fn shared_link_scores(
    anchors: &[f32],
    low_records: &[f32],
    numel: usize,
    total_count: usize,
) -> [f32; 3] {
    if total_count <= SPINE_K {
        return [f32::INFINITY; 3];
    }

    let mut scores = [f32::INFINITY; 3];
    if let Some(curve) = pooled_normalized_curve(anchors, numel, SpineLink::Probit) {
        scores[SpineLink::Probit.score_index()] = straightness_score(&curve, probit_grid());
        scores[SpineLink::Linear.score_index()] = straightness_score(&curve, rank_grid());
    }
    if log_link_is_valid(anchors, low_records, numel)
        && let Some(curve) = pooled_normalized_curve(anchors, numel, SpineLink::LogProbit)
    {
        scores[SpineLink::LogProbit.score_index()] = straightness_score(&curve, probit_grid());
    }
    scores
}

// For an atom-free spine, evaluating the empirical CDF at every anchor gives a
// cheap lower bound on KS distance. It is sufficient for calm batches; a bound
// that reaches the DKW threshold is recomputed exactly at all batch values.
fn ks_distance_screened(
    anchors: &[f32],
    batch: &[f32],
    old_n: usize,
    meta: PositionMeta,
    threshold: f32,
    link: SpineLink,
) -> f32 {
    let secondary_active = meta.secondary_count != 0 && !secondary_candidate(meta.surprise_word);
    if meta.zero_count != 0 || secondary_active {
        return ks_distance_spine(anchors, batch, old_n, meta, link);
    }

    let ranks = rank_grid();
    let inverse_batch_len = 1.0 / batch.len() as f32;
    let mut batch_index = 0usize;
    let mut distance = 0.0f32;
    for (anchor_index, &anchor) in anchors.iter().enumerate() {
        while batch_index < batch.len() && batch[batch_index] < anchor {
            batch_index += 1;
        }
        let before = batch_index as f32 * inverse_batch_len;
        distance = distance.max((ranks[anchor_index] - before).abs());
        while batch_index < batch.len() && batch[batch_index] == anchor {
            batch_index += 1;
        }
        let after = batch_index as f32 * inverse_batch_len;
        distance = distance.max((ranks[anchor_index] - after).abs());
    }

    if distance >= threshold {
        ks_distance_spine(anchors, batch, old_n, meta, link)
    } else {
        distance
    }
}

fn sort_batches_simd<T: TensorValue>(
    row_buffer: &[T],
    numel: usize,
    batch_len: usize,
    output: &mut [f32],
) {
    const LANES: usize = 8;
    let padded_len = batch_len.next_power_of_two();
    output
        .par_chunks_mut(batch_len * LANES)
        .enumerate()
        .for_each_init(
            || vec![f32x8::splat(f32::INFINITY); padded_len],
            |values, (tile, tile_output)| {
                let first_position = tile * LANES;
                let width = tile_output.len() / batch_len;
                if width < LANES {
                    let mut batch = Vec::with_capacity(batch_len);
                    for lane in 0..width {
                        batch.clear();
                        batch.extend((0..batch_len).map(|sample| {
                            row_buffer[sample * numel + first_position + lane].to_f32()
                        }));
                        assert!(
                            batch.iter().all(|value| !value.is_nan()),
                            "QuantileSpine does not accept NaN values"
                        );
                        batch.sort_unstable_by(f32::total_cmp);
                        tile_output[lane * batch_len..(lane + 1) * batch_len]
                            .copy_from_slice(&batch);
                    }
                    return;
                }

                for sample in 0..batch_len {
                    let lanes = std::array::from_fn(|lane| {
                        row_buffer[sample * numel + first_position + lane].to_f32()
                    });
                    assert!(
                        lanes.iter().all(|value| !value.is_nan()),
                        "QuantileSpine does not accept NaN values"
                    );
                    values[sample] = f32x8::from(lanes);
                }
                values[batch_len..].fill(f32x8::splat(f32::INFINITY));
                bitonic_sort(values);

                for (sample, values) in values[..batch_len].iter().enumerate() {
                    for (lane, &value) in values.as_array_ref().iter().enumerate() {
                        tile_output[lane * batch_len + sample] = value;
                    }
                }
            },
        );
}

fn bitonic_sort(values: &mut [f32x8]) {
    debug_assert!(values.len().is_power_of_two());
    let mut sequence_len = 2;
    while sequence_len <= values.len() {
        let mut stride = sequence_len / 2;
        while stride > 0 {
            for index in 0..values.len() {
                let partner = index ^ stride;
                if partner > index {
                    let left = values[index];
                    let right = values[partner];
                    if index & sequence_len == 0 {
                        values[index] = left.min(right);
                        values[partner] = left.max(right);
                    } else {
                        values[index] = left.max(right);
                        values[partner] = left.min(right);
                    }
                }
            }
            stride /= 2;
        }
        sequence_len *= 2;
    }
}

struct FlushScratch {
    batch: Vec<f32>,
    merged: Vec<f32>,
    smooth: Vec<f32>,
    old: [f32; SPINE_K],
    output: [f32; SPINE_K],
    low_records: [f32; RECORDS_T],
    high_records: [f32; RECORDS_T],
    window_low_records: [f32; WINDOW_RECORDS_T],
    window_high_records: [f32; WINDOW_RECORDS_T],
}

impl FlushScratch {
    fn new(batch_len: usize) -> Self {
        Self {
            batch: Vec::with_capacity(batch_len),
            merged: Vec::with_capacity(batch_len + SPINE_K),
            smooth: Vec::with_capacity(batch_len + SPINE_K),
            old: [0.0; SPINE_K],
            output: [0.0; SPINE_K],
            low_records: [0.0; RECORDS_T],
            high_records: [0.0; RECORDS_T],
            window_low_records: [0.0; WINDOW_RECORDS_T],
            window_high_records: [0.0; WINDOW_RECORDS_T],
        }
    }
}

const fn const_link<const LINK: u8>() -> SpineLink {
    match LINK {
        LINEAR_LINK => SpineLink::Linear,
        LOG_PROBIT_LINK => SpineLink::LogProbit,
        _ => SpineLink::Probit,
    }
}

fn quantiles_for_cell<const LINK: u8>(
    anchors: &[f32],
    low: &[f32],
    high: &[f32],
    meta: PositionMeta,
    n: usize,
    qs: &[f32],
) -> Vec<f32> {
    qs.iter()
        .map(|&q| {
            quantile_element::<LINK>(
                anchors,
                low,
                high,
                meta,
                n,
                q,
                query_interpolation::<LINK>(q),
            )
        })
        .collect()
}

#[inline(always)]
fn quantile_element<const LINK: u8>(
    anchors: &[f32],
    low: &[f32],
    high: &[f32],
    meta: PositionMeta,
    n: usize,
    q: f32,
    interpolation: QueryInterpolation,
) -> f32 {
    let link = const_link::<LINK>();
    if n == 0 {
        return 0.0;
    }
    if q.is_nan() {
        return f32::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    if n <= SPINE_K {
        return exact_sorted_quantile(&anchors[..n], q);
    }

    let position = q * (n - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if upper < RECORDS_T {
        return interpolate_order_statistic(low, lower, upper, position);
    }
    if lower >= n - RECORDS_T {
        return interpolate_order_statistic(
            high,
            lower - (n - RECORDS_T),
            upper - (n - RECORDS_T),
            position - (n - RECORDS_T) as f32,
        );
    }

    let zero_count = meta.zero_count as usize;
    let secondary_count = if secondary_candidate(meta.surprise_word) {
        0
    } else {
        meta.secondary_count as usize
    };
    if zero_count == 0 && secondary_count == 0 {
        if let Some(tail) = evt_tail_quantile(anchors, low, high, n, q) {
            return tail;
        }
        return quantile_from_query::<LINK>(anchors, interpolation);
    }

    let smooth_count = n.saturating_sub(zero_count).saturating_sub(secondary_count);
    let mut atoms = [(0.0f32, 0usize); 2];
    let mut atom_len = 0;
    if zero_count > 0 {
        atoms[atom_len] = (0.0, zero_count);
        atom_len += 1;
    }
    if secondary_count > 0 {
        atoms[atom_len] = (meta.secondary_value, secondary_count);
        atom_len += 1;
    }
    if atom_len == 2 && atoms[0].0 > atoms[1].0 {
        atoms.swap(0, 1);
    }

    if smooth_count == 0 {
        let target = q * n as f32;
        let mut cumulative = 0usize;
        for &(value, count) in &atoms[..atom_len] {
            cumulative += count;
            if target <= cumulative as f32 {
                return value;
            }
        }
        return atoms[atom_len - 1].0;
    }

    let smooth_mass = smooth_count as f32 / n as f32;
    let mut prior_atom_mass = 0.0f32;
    let mut i = 0;
    while i < atom_len {
        let value = atoms[i].0;
        let mut count = atoms[i].1;
        i += 1;
        while i < atom_len && atoms[i].0 == value {
            count += atoms[i].1;
            i += 1;
        }
        let smooth_before = cdf_from_anchors(anchors, value, false, link);
        let full_before = prior_atom_mass + smooth_mass * smooth_before;
        if q < full_before {
            return quantile_from_anchors_for::<LINK>(
                anchors,
                ((q - prior_atom_mass) / smooth_mass).clamp(0.0, 1.0),
            );
        }
        let atom_mass = count as f32 / n as f32;
        if q <= full_before + atom_mass {
            return value;
        }
        prior_atom_mass += atom_mass;
    }

    quantile_from_anchors_for::<LINK>(
        anchors,
        ((q - prior_atom_mass) / smooth_mass).clamp(0.0, 1.0),
    )
}

fn gpd_factor(tail_ratio: f32, shape: f32) -> f32 {
    if shape.abs() < 1.0e-4 {
        -f32::ln(tail_ratio)
    } else {
        f32::exp_m1(-shape * f32::ln(tail_ratio)) / shape
    }
}

fn evt_tail_quantile(anchors: &[f32], low: &[f32], high: &[f32], n: usize, q: f32) -> Option<f32> {
    const OUTER: usize = 1;
    const THRESHOLD: usize = 2;
    let outer_probability = rank_grid()[OUTER];
    let shelf_probability = (RECORDS_T - 1) as f32 / (n - 1) as f32;
    let tail_probability = if q < outer_probability {
        q
    } else if 1.0 - q < outer_probability {
        1.0 - q
    } else {
        return None;
    };
    if tail_probability <= shelf_probability || shelf_probability >= outer_probability {
        return None;
    }

    let lower_tail = q < 0.5;
    let (threshold_value, outer_value, shelf_value) = if lower_tail {
        (anchors[THRESHOLD], anchors[OUTER], low[RECORDS_T - 1])
    } else {
        (
            anchors[SPINE_K - 1 - THRESHOLD],
            anchors[SPINE_K - 1 - OUTER],
            high[0],
        )
    };
    let outer_excess = if lower_tail {
        threshold_value - outer_value
    } else {
        outer_value - threshold_value
    };
    let shelf_excess = if lower_tail {
        threshold_value - shelf_value
    } else {
        shelf_value - threshold_value
    };
    if !(outer_excess > 0.0 && shelf_excess > outer_excess) {
        return None;
    }

    let threshold_probability = rank_grid()[THRESHOLD];
    let outer_ratio = outer_probability / threshold_probability;
    let shelf_ratio = shelf_probability / threshold_probability;
    let observed_ratio = shelf_excess / outer_excess;
    let modeled_ratio = |shape| gpd_factor(shelf_ratio, shape) / gpd_factor(outer_ratio, shape);
    let mut shape_lower = -0.5f32;
    let mut shape_upper = 0.5f32;
    for _ in 0..20 {
        let middle = 0.5 * (shape_lower + shape_upper);
        if modeled_ratio(middle) < observed_ratio {
            shape_lower = middle;
        } else {
            shape_upper = middle;
        }
    }
    let shape = 0.5 * (shape_lower + shape_upper);
    let excess = outer_excess * gpd_factor(tail_probability / threshold_probability, shape)
        / gpd_factor(outer_ratio, shape);
    if !excess.is_finite() || excess < outer_excess {
        return None;
    }
    let estimate = if lower_tail {
        threshold_value - excess
    } else {
        threshold_value + excess
    };
    Some(if lower_tail {
        estimate.clamp(shelf_value, outer_value)
    } else {
        estimate.clamp(outer_value, shelf_value)
    })
}

fn interpolate_order_statistic(values: &[f32], lower: usize, upper: usize, position: f32) -> f32 {
    if lower == upper {
        values[lower]
    } else {
        let t = position - position.floor();
        values[lower] + t * (values[upper] - values[lower])
    }
}

fn exact_sorted_quantile(values: &[f32], q: f32) -> f32 {
    if values.len() == 1 {
        return values[0];
    }
    let position = q * (values.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    interpolate_order_statistic(values, lower, upper, position)
}

fn rank_grid() -> &'static [f32; SPINE_K] {
    static GRID: OnceLock<[f32; SPINE_K]> = OnceLock::new();
    GRID.get_or_init(|| {
        let mut grid = [0.0; SPINE_K];
        for j in 0..SPINE_K / 2 {
            let angle = std::f32::consts::PI * j as f32 / (2 * (SPINE_K - 1)) as f32;
            let sine = f32::sin(angle);
            let rank = sine * sine;
            grid[j] = rank;
            grid[SPINE_K - 1 - j] = 1.0 - rank;
        }
        if SPINE_K % 2 != 0 {
            grid[SPINE_K / 2] = 0.5;
        }
        grid
    })
}

fn probit_grid() -> &'static [f32; SPINE_K] {
    static GRID: OnceLock<[f32; SPINE_K]> = OnceLock::new();
    GRID.get_or_init(|| rank_grid().map(probit))
}

fn shifted_ranks(eta: f32) -> [f32; SPINE_K] {
    let ranks = rank_grid();
    std::array::from_fn(|j| {
        let index = (j as f32 + eta).clamp(0.0, (SPINE_K - 1) as f32);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        if lower == upper {
            ranks[lower]
        } else {
            ranks[lower] + (index - lower as f32) * (ranks[upper] - ranks[lower])
        }
    })
}

#[derive(Clone, Copy)]
struct SampleInterpolation {
    left: usize,
    right: usize,
    fraction: f32,
}

fn plotting_position_adjustment(link: SpineLink) -> f32 {
    match link {
        SpineLink::Linear => 0.0,
        SpineLink::Probit | SpineLink::LogProbit => 3.0 / 8.0,
    }
}

fn plotting_rank(index: usize, sample_count: usize, link: SpineLink) -> f32 {
    let adjustment = plotting_position_adjustment(link);
    (index as f32 + 1.0 - adjustment) / (sample_count as f32 + 1.0 - 2.0 * adjustment)
}

fn sample_interpolation(
    targets: &[f32; SPINE_K],
    sample_count: usize,
    link: SpineLink,
) -> [SampleInterpolation; SPINE_K] {
    std::array::from_fn(|target_index| {
        if sample_count <= 1 {
            return SampleInterpolation {
                left: 0,
                right: 0,
                fraction: 0.0,
            };
        }
        let q = targets[target_index];
        let adjustment = plotting_position_adjustment(link);
        let position = q * (sample_count as f32 + 1.0 - 2.0 * adjustment) - (1.0 - adjustment);
        let left = (position.floor() as isize).clamp(0, sample_count as isize - 2) as usize;
        let right = left + 1;
        let x0 = link.rank_coordinate(plotting_rank(left, sample_count, link));
        let x1 = link.rank_coordinate(plotting_rank(right, sample_count, link));
        SampleInterpolation {
            left,
            right,
            fraction: (link.rank_coordinate(q) - x0) / (x1 - x0),
        }
    })
}

fn interpolated_sample_quantile<const LOG: bool>(
    samples: &[f32],
    interpolation: SampleInterpolation,
) -> f32 {
    let left = samples[interpolation.left];
    let right = samples[interpolation.right];
    if LOG && (left <= 0.0 || right <= 0.0) {
        return left + interpolation.fraction * (right - left);
    }
    let y0 = scale_value::<LOG>(left);
    let y1 = scale_value::<LOG>(right);
    unscale_value::<LOG>(y0 + interpolation.fraction * (y1 - y0))
}

#[inline(always)]
fn scale_value<const LOG: bool>(value: f32) -> f32 {
    if LOG { value.ln() } else { value }
}

#[inline(always)]
fn unscale_value<const LOG: bool>(value: f32) -> f32 {
    if LOG { value.exp() } else { value }
}

fn monotone_tangent<const LOG: bool>(anchors: &[f32], axis: &[f32; SPINE_K], index: usize) -> f32 {
    let secant = |left: usize| {
        let width = axis[left + 1] - axis[left];
        if width <= 0.0 {
            0.0
        } else {
            (scale_value::<LOG>(anchors[left + 1]) - scale_value::<LOG>(anchors[left])) / width
        }
    };

    if index == 0 {
        let h0 = axis[1] - axis[0];
        let h1 = axis[2] - axis[1];
        let d0 = secant(0);
        let d1 = secant(1);
        if d0 == 0.0 {
            return 0.0;
        }
        let tangent = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if tangent.signum() != d0.signum() {
            0.0
        } else if d0.signum() != d1.signum() && tangent.abs() > 3.0 * d0.abs() {
            3.0 * d0
        } else {
            tangent
        }
    } else if index == SPINE_K - 1 {
        let h0 = axis[SPINE_K - 1] - axis[SPINE_K - 2];
        let h1 = axis[SPINE_K - 2] - axis[SPINE_K - 3];
        let d0 = secant(SPINE_K - 2);
        let d1 = secant(SPINE_K - 3);
        if d0 == 0.0 {
            return 0.0;
        }
        let tangent = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if tangent.signum() != d0.signum() {
            0.0
        } else if d0.signum() != d1.signum() && tangent.abs() > 3.0 * d0.abs() {
            3.0 * d0
        } else {
            tangent
        }
    } else {
        let previous = secant(index - 1);
        let next = secant(index);
        if previous == 0.0 || next == 0.0 || previous.signum() != next.signum() {
            return 0.0;
        }
        let h_previous = axis[index] - axis[index - 1];
        let h_next = axis[index + 1] - axis[index];
        let w1 = 2.0 * h_next + h_previous;
        let w2 = h_next + 2.0 * h_previous;
        (w1 + w2) / (w1 / previous + w2 / next)
    }
}

#[inline(always)]
fn interior_monotone_tangent(
    previous: f32,
    next: f32,
    previous_width: f32,
    next_width: f32,
) -> f32 {
    if previous == 0.0 || next == 0.0 || previous.signum() != next.signum() {
        return 0.0;
    }
    let w1 = 2.0 * next_width + previous_width;
    let w2 = next_width + 2.0 * previous_width;
    (w1 + w2) / (w1 / previous + w2 / next)
}

fn cubic_segment<const LOG: bool>(
    anchors: &[f32],
    axis: &[f32; SPINE_K],
    left: usize,
    t: f32,
) -> (f32, f32) {
    let right = left + 1;
    let width = axis[right] - axis[left];
    let y0 = scale_value::<LOG>(anchors[left]);
    let y1 = scale_value::<LOG>(anchors[right]);
    if width <= 0.0 || y1 <= y0 {
        return (y0, 0.0);
    }
    let (m0, m1) = if left > 0 && right + 1 < SPINE_K {
        let previous_width = axis[left] - axis[left - 1];
        let next_width = axis[right + 1] - axis[right];
        let previous = (y0 - scale_value::<LOG>(anchors[left - 1])) / previous_width;
        let middle = (y1 - y0) / width;
        let next = (scale_value::<LOG>(anchors[right + 1]) - y1) / next_width;
        (
            interior_monotone_tangent(previous, middle, previous_width, width) * width,
            interior_monotone_tangent(middle, next, width, next_width) * width,
        )
    } else {
        (
            monotone_tangent::<LOG>(anchors, axis, left) * width,
            monotone_tangent::<LOG>(anchors, axis, right) * width,
        )
    };
    let t2 = t * t;
    let t3 = t2 * t;
    let value = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * m1;
    let derivative = (6.0 * t2 - 6.0 * t) * y0
        + (3.0 * t2 - 4.0 * t + 1.0) * m0
        + (-6.0 * t2 + 6.0 * t) * y1
        + (3.0 * t2 - 2.0 * t) * m1;
    (value.clamp(y0, y1), derivative.max(0.0))
}

#[derive(Clone, Copy)]
struct QueryInterpolation {
    lower: usize,
    fraction: f32,
}

fn query_interpolation<const LINK: u8>(q: f32) -> QueryInterpolation {
    let (axis, coordinate) = match LINK {
        LINEAR_LINK => (rank_grid(), q),
        _ => (probit_grid(), probit(q)),
    };
    let upper = rank_grid()
        .partition_point(|&rank| rank < q)
        .clamp(1, SPINE_K - 1);
    let lower = upper - 1;
    QueryInterpolation {
        lower,
        fraction: ((coordinate - axis[lower]) / (axis[upper] - axis[lower])).clamp(0.0, 1.0),
    }
}

#[inline(always)]
fn quantile_from_query<const LINK: u8>(anchors: &[f32], interpolation: QueryInterpolation) -> f32 {
    if LINK == LINEAR_LINK {
        let left = interpolation.lower;
        anchors[left] + interpolation.fraction * (anchors[left + 1] - anchors[left])
    } else if LINK == LOG_PROBIT_LINK {
        unscale_value::<true>(
            cubic_segment::<true>(
                anchors,
                probit_grid(),
                interpolation.lower,
                interpolation.fraction,
            )
            .0,
        )
    } else {
        cubic_segment::<false>(
            anchors,
            probit_grid(),
            interpolation.lower,
            interpolation.fraction,
        )
        .0
    }
}

#[inline(always)]
fn quantile_from_anchors_for<const LINK: u8>(anchors: &[f32], q: f32) -> f32 {
    if q <= 0.0 {
        return anchors[0];
    }
    if q >= 1.0 {
        return anchors[SPINE_K - 1];
    }
    quantile_from_query::<LINK>(anchors, query_interpolation::<LINK>(q))
}

#[cfg(test)]
fn quantile_from_anchors(anchors: &[f32], q: f32, link: SpineLink) -> f32 {
    match link {
        SpineLink::Probit => quantile_from_anchors_for::<PROBIT_LINK>(anchors, q),
        SpineLink::Linear => quantile_from_anchors_for::<LINEAR_LINK>(anchors, q),
        SpineLink::LogProbit => quantile_from_anchors_for::<LOG_PROBIT_LINK>(anchors, q),
    }
}

fn quantile_from_grid_cursor(
    anchors: &[f32],
    grid: &[f32; SPINE_K],
    axis_grid: &[f32; SPINE_K],
    q: f32,
    cursor: &mut usize,
    link: SpineLink,
) -> f32 {
    if q <= grid[0] {
        return anchors[0];
    }
    if q >= grid[SPINE_K - 1] {
        return anchors[SPINE_K - 1];
    }
    while *cursor + 1 < SPINE_K && grid[*cursor + 1] < q {
        *cursor += 1;
    }
    let upper = (*cursor + 1).min(SPINE_K - 1);
    let x0 = axis_grid[*cursor];
    let x1 = axis_grid[upper];
    let denominator = x1 - x0;
    if denominator <= 0.0 {
        anchors[*cursor]
    } else {
        let t = ((link.rank_coordinate(q) - x0) / denominator).clamp(0.0, 1.0);
        let y0 = link.value_coordinate(anchors[*cursor]);
        let y1 = link.value_coordinate(anchors[upper]);
        link.value_from_coordinate(y0 + t * (y1 - y0))
    }
}

fn cdf_from_anchors(anchors: &[f32], value: f32, inclusive: bool, link: SpineLink) -> f32 {
    let first = if inclusive {
        anchors.partition_point(|&anchor| anchor <= value)
    } else {
        anchors.partition_point(|&anchor| anchor < value)
    };
    if first == 0 {
        return 0.0;
    }
    if first == SPINE_K {
        return 1.0;
    }
    let left = first - 1;
    if anchors[left] == value {
        return rank_grid()[left];
    }
    interpolate_anchor_cdf_cubic(anchors, left, value, link)
}

fn interpolate_anchor_cdf_cubic(anchors: &[f32], left: usize, value: f32, link: SpineLink) -> f32 {
    let right = left + 1;
    if anchors[right] <= anchors[left] {
        return rank_grid()[right];
    }
    if link == SpineLink::Linear {
        return interpolate_anchor_cdf(anchors, left, right, value, link);
    }
    let axis = link.transformed_grid();
    let target = link.value_coordinate(value);
    let y0 = link.value_coordinate(anchors[left]);
    let y1 = link.value_coordinate(anchors[right]);
    let mut lower = 0.0f32;
    let mut upper = 1.0f32;
    let mut t = ((target - y0) / (y1 - y0)).clamp(0.0, 1.0);
    for _ in 0..5 {
        let (estimate, derivative) = match link {
            SpineLink::LogProbit => cubic_segment::<true>(anchors, axis, left, t),
            SpineLink::Probit | SpineLink::Linear => cubic_segment::<false>(anchors, axis, left, t),
        };
        if estimate < target {
            lower = t;
        } else {
            upper = t;
        }
        let candidate = if derivative > f32::EPSILON {
            t - (estimate - target) / derivative
        } else {
            f32::NAN
        };
        t = if candidate > lower && candidate < upper {
            candidate
        } else {
            0.5 * (lower + upper)
        };
    }
    link.rank_from_coordinate(axis[left] + t * (axis[right] - axis[left]))
}

fn interpolate_anchor_cdf(
    anchors: &[f32],
    left: usize,
    right: usize,
    value: f32,
    link: SpineLink,
) -> f32 {
    interpolate_grid_cdf(
        anchors,
        rank_grid(),
        link.transformed_grid(),
        left,
        right,
        value,
        link,
    )
}

fn interpolate_grid_cdf(
    anchors: &[f32],
    grid: &[f32; SPINE_K],
    axis_grid: &[f32; SPINE_K],
    left: usize,
    right: usize,
    value: f32,
    link: SpineLink,
) -> f32 {
    if anchors[right] <= anchors[left] {
        return grid[right];
    }
    let y0 = link.value_coordinate(anchors[left]);
    let y1 = link.value_coordinate(anchors[right]);
    let t = ((link.value_coordinate(value) - y0) / (y1 - y0)).clamp(0.0, 1.0);
    let coordinate = axis_grid[left] + t * (axis_grid[right] - axis_grid[left]);
    link.rank_from_coordinate(coordinate)
}

fn empirical_reanchor(values: &[f32], targets: &[f32; SPINE_K], output: &mut [f32; SPINE_K]) {
    if values.is_empty() {
        output.fill(0.0);
        return;
    }
    for (out, &q) in output.iter_mut().zip(targets) {
        let index = if q <= 0.0 {
            0
        } else {
            ((q * values.len() as f32).ceil() as usize)
                .saturating_sub(1)
                .min(values.len() - 1)
        };
        *out = values[index];
    }
    enforce_monotone(output);
}

// Fast path for distributions that passed the KS/DKW closeness screen.
fn blend_reanchor_close_scaled<const LOG: bool>(
    old: &[f32; SPINE_K],
    batch: &[f32],
    new_gain: f32,
    targets: &[f32; SPINE_K],
    batch_interpolation: &[SampleInterpolation; SPINE_K],
    output: &mut [f32; SPINE_K],
) {
    if batch.is_empty() {
        output.copy_from_slice(old);
        return;
    }
    if new_gain <= f32::EPSILON {
        output.copy_from_slice(old);
        return;
    }

    let history_weight = 1.0 - new_gain;
    let support_min = old[0].min(batch[0]);
    let support_max = old[SPINE_K - 1].max(batch[batch.len() - 1]);
    for (index, &q) in targets.iter().enumerate() {
        if q <= 0.0 {
            output[index] = support_min;
            continue;
        }
        if q >= 1.0 {
            output[index] = support_max;
            continue;
        }
        let batch_value = interpolated_sample_quantile::<LOG>(batch, batch_interpolation[index])
            .clamp(support_min, support_max);
        output[index] = if new_gain >= 1.0 - f32::EPSILON {
            batch_value
        } else {
            new_gain.mul_add(batch_value, history_weight * old[index])
        };
    }
    enforce_monotone(output);
}

fn blend_reanchor_close(
    old: &[f32; SPINE_K],
    batch: &[f32],
    new_gain: f32,
    targets: &[f32; SPINE_K],
    batch_interpolation: &[SampleInterpolation; SPINE_K],
    link: SpineLink,
    output: &mut [f32; SPINE_K],
) {
    match link {
        SpineLink::LogProbit => blend_reanchor_close_scaled::<true>(
            old,
            batch,
            new_gain,
            targets,
            batch_interpolation,
            output,
        ),
        SpineLink::Probit | SpineLink::Linear => blend_reanchor_close_scaled::<false>(
            old,
            batch,
            new_gain,
            targets,
            batch_interpolation,
            output,
        ),
    }
}

fn blend_reanchor(
    old: &[f32; SPINE_K],
    batch: &[f32],
    new_gain: f32,
    targets: &[f32; SPINE_K],
    target_axis: &[f32; SPINE_K],
    link: SpineLink,
    output: &mut [f32; SPINE_K],
) {
    if batch.is_empty() {
        output.copy_from_slice(old);
        return;
    }
    if new_gain >= 1.0 - f32::EPSILON {
        empirical_reanchor(batch, targets, output);
        return;
    }
    if new_gain <= f32::EPSILON {
        output.copy_from_slice(old);
        return;
    }

    // Shift the old ruler and the requested ruler together. Thus a history-only
    // merge is exactly the identity; randomness affects only the interpolation
    // residual instead of introducing a first-order rank random walk.
    let history_weight = 1.0 - new_gain;
    let support_min = old[0].min(batch[0]);
    let support_max = old[SPINE_K - 1].max(batch[batch.len() - 1]);
    let ranks = targets;
    let mut old_index = 0usize;
    let mut batch_index = 0usize;
    let mut target_index = 0usize;
    let mut quantile_cursor = 0usize;

    while target_index < SPINE_K && targets[target_index] <= 0.0 {
        output[target_index] = support_min;
        target_index += 1;
    }

    while old_index < SPINE_K || batch_index < batch.len() {
        let old_value = old.get(old_index).copied().unwrap_or(f32::INFINITY);
        let batch_value = batch.get(batch_index).copied().unwrap_or(f32::INFINITY);
        let event = old_value.min(batch_value);

        let old_before = if old_index == 0 {
            0.0
        } else if old_index >= SPINE_K {
            1.0
        } else if old_value == event {
            ranks[old_index]
        } else {
            interpolate_grid_cdf(
                old,
                ranks,
                target_axis,
                old_index - 1,
                old_index,
                event,
                link,
            )
        };
        let batch_before = batch_index as f32 / batch.len() as f32;
        let mixture_before = history_weight * old_before + new_gain * batch_before;

        while target_index < SPINE_K && targets[target_index] <= mixture_before {
            let old_rank = ((targets[target_index] - new_gain * batch_before) / history_weight)
                .clamp(0.0, 1.0);
            output[target_index] = quantile_from_grid_cursor(
                old,
                ranks,
                target_axis,
                old_rank,
                &mut quantile_cursor,
                link,
            );
            target_index += 1;
        }

        let mut next_old = old_index;
        while next_old < SPINE_K && old[next_old] == event {
            next_old += 1;
        }
        let old_after = if next_old == SPINE_K {
            1.0
        } else if next_old > old_index {
            ranks[next_old - 1]
        } else {
            old_before
        };

        let mut next_batch = batch_index;
        while next_batch < batch.len() && batch[next_batch] == event {
            next_batch += 1;
        }
        let batch_after = next_batch as f32 / batch.len() as f32;
        let mixture_after = history_weight * old_after + new_gain * batch_after;
        while target_index < SPINE_K && targets[target_index] <= mixture_after {
            output[target_index] = event;
            target_index += 1;
        }

        old_index = next_old;
        batch_index = next_batch;
    }

    output[target_index..].fill(support_max);
    enforce_monotone(output);
}

fn enforce_monotone(values: &mut [f32]) {
    for i in 1..values.len() {
        if values[i] < values[i - 1] {
            values[i] = values[i - 1];
        }
    }
}

fn ks_distance_exact(old: &[f32], batch: &[f32], old_n: f32) -> f32 {
    let mut old_index = 0usize;
    let mut batch_index = 0usize;
    let mut distance = 0.0f32;
    while batch_index < batch.len() {
        let value = batch[batch_index];
        while old_index < old.len() && old[old_index] < value {
            old_index += 1;
        }
        let old_before = old_index as f32 / old_n;
        let batch_before = batch_index as f32 / batch.len() as f32;
        distance = distance.max((old_before - batch_before).abs());

        while old_index < old.len() && old[old_index] <= value {
            old_index += 1;
        }
        let mut next_batch = batch_index + 1;
        while next_batch < batch.len() && batch[next_batch] == value {
            next_batch += 1;
        }
        let old_after = old_index as f32 / old_n;
        let batch_after = next_batch as f32 / batch.len() as f32;
        distance = distance.max((old_after - batch_after).abs());
        batch_index = next_batch;
    }
    distance
}

fn ks_distance_spine(
    anchors: &[f32],
    batch: &[f32],
    old_n: usize,
    meta: PositionMeta,
    link: SpineLink,
) -> f32 {
    let secondary_count = if secondary_candidate(meta.surprise_word) {
        0
    } else {
        meta.secondary_count
    };
    let zero_mass = meta.zero_count as f32 / old_n as f32;
    let secondary_mass = secondary_count as f32 / old_n as f32;
    let smooth_mass = (old_n
        .saturating_sub(meta.zero_count as usize)
        .saturating_sub(secondary_count as usize)) as f32
        / old_n as f32;
    let mut anchor_index = 0usize;
    let mut batch_index = 0usize;
    let mut distance = 0.0f32;
    while batch_index < batch.len() {
        let value = batch[batch_index];
        while anchor_index < SPINE_K && anchors[anchor_index] < value {
            anchor_index += 1;
        }
        let smooth_before = cdf_at_anchor_cursor(anchors, anchor_index, value, false, link);
        let mut old_before = smooth_mass * smooth_before;
        if value > 0.0 {
            old_before += zero_mass;
        }
        if secondary_count > 0 && value > meta.secondary_value {
            old_before += secondary_mass;
        }
        let batch_before = batch_index as f32 / batch.len() as f32;
        distance = distance.max((old_before - batch_before).abs());

        while anchor_index < SPINE_K && anchors[anchor_index] <= value {
            anchor_index += 1;
        }
        let smooth_after = cdf_at_anchor_cursor(anchors, anchor_index, value, true, link);
        let mut old_after = smooth_mass * smooth_after;
        if value >= 0.0 {
            old_after += zero_mass;
        }
        if secondary_count > 0 && value >= meta.secondary_value {
            old_after += secondary_mass;
        }
        let mut next_batch = batch_index + 1;
        while next_batch < batch.len() && batch[next_batch] == value {
            next_batch += 1;
        }
        let batch_after = next_batch as f32 / batch.len() as f32;
        distance = distance.max((old_after - batch_after).abs());
        batch_index = next_batch;
    }
    distance
}

fn cdf_at_anchor_cursor(
    anchors: &[f32],
    index: usize,
    value: f32,
    inclusive: bool,
    link: SpineLink,
) -> f32 {
    if index == 0 {
        return 0.0;
    }
    if index == SPINE_K {
        return 1.0;
    }
    let left = index - 1;
    if anchors[left] == value {
        return rank_grid()[left];
    }
    if !inclusive && anchors[index] == value {
        return rank_grid()[index];
    }
    interpolate_anchor_cdf(anchors, left, index, value, link)
}

fn update_surprise(
    word: &mut u32,
    distance: f32,
    threshold: f32,
    config: QuantileSpineConfig,
) -> (bool, bool) {
    let updated_gauge =
        (1.0 - config.gauge_rho) * gauge(*word) + config.gauge_rho * distance.clamp(0.0, 1.0);
    set_gauge(word, updated_gauge);

    let mut restarted = false;
    let mut suppress = false;
    let remaining = suppression(*word);
    if remaining > 0 {
        suppress = true;
        let next = remaining - 1;
        set_suppression(word, next);
        if next == 0 {
            set_regime(word, SpineRegime::Calm);
        } else {
            set_regime(word, SpineRegime::Restart);
        }
        set_crossings(word, 0);
        return (false, suppress);
    }

    if distance > threshold {
        let count = (crossings(*word) + 1).min(3);
        set_crossings(word, count);
        if config.restart_crossings > 0 && count >= config.restart_crossings {
            restarted = true;
            suppress = true;
            set_regime(word, SpineRegime::Restart);
            set_crossings(word, 0);
            set_suppression(word, config.restart_batches.saturating_sub(1));
        } else {
            set_regime(word, SpineRegime::Alert);
        }
    } else {
        set_crossings(word, 0);
        set_regime(word, SpineRegime::Calm);
    }
    (restarted, suppress)
}

fn gauge(word: u32) -> f32 {
    (word & GAUGE_MASK) as f32 / u16::MAX as f32
}

fn set_gauge(word: &mut u32, value: f32) {
    let quantized = (value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u32;
    *word = (*word & !GAUGE_MASK) | quantized;
}

fn regime(word: u32) -> SpineRegime {
    match (word & REGIME_MASK) >> REGIME_SHIFT {
        1 => SpineRegime::Alert,
        2 => SpineRegime::Restart,
        _ => SpineRegime::Calm,
    }
}

fn set_regime(word: &mut u32, value: SpineRegime) {
    let bits = match value {
        SpineRegime::Calm => 0,
        SpineRegime::Alert => 1,
        SpineRegime::Restart => 2,
    };
    *word = (*word & !REGIME_MASK) | (bits << REGIME_SHIFT);
}

fn crossings(word: u32) -> u8 {
    ((word & CROSSINGS_MASK) >> CROSSINGS_SHIFT) as u8
}

fn set_crossings(word: &mut u32, value: u8) {
    *word = (*word & !CROSSINGS_MASK) | ((value.min(3) as u32) << CROSSINGS_SHIFT);
}

fn suppression(word: u32) -> u8 {
    ((word & SUPPRESSION_MASK) >> SUPPRESSION_SHIFT) as u8
}

fn set_suppression(word: &mut u32, value: u8) {
    *word = (*word & !SUPPRESSION_MASK) | ((value.min(7) as u32) << SUPPRESSION_SHIFT);
}

fn window_count(word: u32) -> usize {
    ((word & WINDOW_COUNT_MASK) >> WINDOW_COUNT_SHIFT) as usize
}

fn set_window_count(word: &mut u32, value: usize) {
    *word = (*word & !WINDOW_COUNT_MASK)
        | (((value.min(WINDOW_RECORDS_T) as u32) << WINDOW_COUNT_SHIFT) & WINDOW_COUNT_MASK);
}

fn secondary_candidate(word: u32) -> bool {
    word & SECONDARY_CANDIDATE_MASK != 0
}

fn set_secondary_candidate(word: &mut u32, candidate: bool) {
    if candidate {
        *word |= SECONDARY_CANDIDATE_MASK;
    } else {
        *word &= !SECONDARY_CANDIDATE_MASK;
    }
}

fn merge_sorted(left: &[f32], right: &[f32], output: &mut Vec<f32>) {
    output.clear();
    output.reserve(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            output.push(left[i]);
            i += 1;
        } else {
            output.push(right[j]);
            j += 1;
        }
    }
    output.extend_from_slice(&left[i..]);
    output.extend_from_slice(&right[j..]);
}

fn merge_low<const CAPACITY: usize>(old: &[f32], batch: &[f32], output: &mut [f32; CAPACITY]) {
    let mut old_index = 0;
    let mut batch_index = 0;
    for slot in output.iter_mut() {
        if old_index >= old.len() && batch_index >= batch.len() {
            *slot = 0.0;
        } else if batch_index >= batch.len()
            || (old_index < old.len() && old[old_index] <= batch[batch_index])
        {
            *slot = old[old_index];
            old_index += 1;
        } else {
            *slot = batch[batch_index];
            batch_index += 1;
        }
    }
}

fn merge_high<const CAPACITY: usize>(old: &[f32], batch: &[f32], output: &mut [f32; CAPACITY]) {
    let count = (old.len() + batch.len()).min(CAPACITY);
    output.fill(0.0);
    let mut old_index = old.len();
    let mut batch_index = batch.len();
    for slot in (0..count).rev() {
        if old_index == 0 {
            batch_index -= 1;
            output[slot] = batch[batch_index];
        } else if batch_index == 0 || old[old_index - 1] >= batch[batch_index - 1] {
            old_index -= 1;
            output[slot] = old[old_index];
        } else {
            batch_index -= 1;
            output[slot] = batch[batch_index];
        }
    }
}

fn find_modal_tie(values: &[f32], threshold: f32, denominator: usize) -> Option<(f32, usize)> {
    let required = ((threshold * denominator as f32).ceil() as usize).max(2);
    let mut best = None;
    let mut i = 0;
    while i < values.len() {
        let value = values[i];
        let mut j = i + 1;
        while j < values.len() && values[j] == value {
            j += 1;
        }
        let count = j - i;
        if value != 0.0
            && count >= required
            && best.is_none_or(|(_, best_count)| count > best_count)
        {
            best = Some((value, count));
        }
        i = j;
    }
    best
}

fn normal_cdf(value: f32) -> f32 {
    const Z_LIMIT: f32 = 5.0;
    const TABLE_POINTS: usize = 4_097;
    static TABLE: OnceLock<[f32; TABLE_POINTS]> = OnceLock::new();

    if value <= -Z_LIMIT {
        return 0.0;
    }
    if value >= Z_LIMIT {
        return 1.0;
    }
    let table = TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            let z = -Z_LIMIT + 2.0 * Z_LIMIT * index as f32 / (TABLE_POINTS - 1) as f32;
            normal_cdf_exact(z)
        })
    });
    let coordinate = (value + Z_LIMIT) * (TABLE_POINTS - 1) as f32 / (2.0 * Z_LIMIT);
    let lower = coordinate.floor() as usize;
    let upper = (lower + 1).min(TABLE_POINTS - 1);
    let fraction = coordinate - lower as f32;
    table[lower] + fraction * (table[upper] - table[lower])
}

fn normal_cdf_exact(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() / std::f32::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    let erf = sign * (1.0 - polynomial * f32::exp(-x * x));
    (0.5 * (1.0 + erf)).clamp(0.0, 1.0)
}

// Peter J. Acklam's inverse-normal approximation. The fixed clipping is the
// finite representation of q=0 and q=1 on the shared probit axis.
fn probit(probability: f32) -> f32 {
    let p = probability.clamp(PROBIT_EPSILON, 1.0 - PROBIT_EPSILON) as f64;
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;

    let value = if p < LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    value as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probit_round_trip() {
        for &q in &[0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999] {
            assert!((normal_cdf(probit(q)) - q).abs() < 2.0e-5);
        }
    }

    #[test]
    fn rank_grid_is_monotone_and_exactly_symmetric() {
        let grid = rank_grid();
        assert_eq!(grid[0], 0.0);
        assert_eq!(grid[SPINE_K - 1], 1.0);
        assert!(grid.windows(2).all(|pair| pair[0] < pair[1]));
        for j in 0..SPINE_K / 2 {
            assert_eq!(grid[SPINE_K - 1 - j], 1.0 - grid[j]);
        }
    }

    #[test]
    fn unified_plotting_positions_match_named_forms() {
        let sample_count = 100;
        for index in 0..sample_count {
            assert_eq!(
                plotting_rank(index, sample_count, SpineLink::Linear),
                (index + 1) as f32 / (sample_count + 1) as f32
            );
            let blom = (index as f32 + 0.625) / (sample_count as f32 + 0.25);
            assert_eq!(plotting_rank(index, sample_count, SpineLink::Probit), blom);
            assert_eq!(
                plotting_rank(index, sample_count, SpineLink::LogProbit),
                blom
            );
        }
    }

    #[test]
    fn cell_quantiles_match_bulk_queries_for_every_link() {
        let config = QuantileSpineConfig {
            link_refit_interval: 0,
            ..QuantileSpineConfig::default()
        };
        let mut spine = QuantileSpineStorage::with_config(&[2], config);
        for index in 0..1_000 {
            let value = index as f32 + 1.0;
            spine.update(&[value, 2.0 * value + 1.0]);
        }
        let qs = [0.001, 0.1, 0.5, 0.9, 0.999];
        for link in [SpineLink::Linear, SpineLink::Probit, SpineLink::LogProbit] {
            spine.link = link;
            let bulk = spine.quantiles(&qs);
            for position in 0..spine.numel() {
                let cell = spine.cell_quantiles(position, &qs);
                for (q_index, &value) in cell.iter().enumerate() {
                    assert_eq!(value, bulk[q_index][position]);
                }
            }
        }
    }

    #[test]
    fn simd_sort_matches_scalar_sort_with_remainder() {
        const NUMEL: usize = 11;
        const BATCH: usize = 13;
        let input = (0..NUMEL * BATCH)
            .map(|index| ((index * 37 + index / 7) % 101) as f32 - 50.0)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        sort_batches_simd(&input, NUMEL, BATCH, &mut output);
        for position in 0..NUMEL {
            let mut expected = (0..BATCH)
                .map(|sample| input[sample * NUMEL + position])
                .collect::<Vec<_>>();
            expected.sort_unstable_by(f32::total_cmp);
            assert_eq!(&output[position * BATCH..(position + 1) * BATCH], expected);
        }
    }

    #[test]
    fn monotone_cubic_does_not_overshoot() {
        let anchors = probit_grid().map(|z| f32::exp(0.2 * z));
        let values = (0..=1_000)
            .map(|index| quantile_from_anchors(&anchors, index as f32 / 1_000.0, SpineLink::Probit))
            .collect::<Vec<_>>();
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(
            values
                .iter()
                .all(|&value| { value >= anchors[0] && value <= anchors[SPINE_K - 1] })
        );
    }

    #[test]
    fn evt_fit_bridges_records_and_outer_anchors() {
        let n = 100_000usize;
        let anchors = rank_grid().map(probit);
        let low: [f32; RECORDS_T] =
            std::array::from_fn(|index| probit(index as f32 / (n - 1) as f32));
        let high: [f32; RECORDS_T] =
            std::array::from_fn(|index| probit((n - RECORDS_T + index) as f32 / (n - 1) as f32));
        let shelf_probability = (RECORDS_T - 1) as f32 / (n - 1) as f32;
        let q = 0.5 * (shelf_probability + rank_grid()[1]);
        let lower = evt_tail_quantile(&anchors, &low, &high, n, q).unwrap();
        let upper = evt_tail_quantile(&anchors, &low, &high, n, 1.0 - q).unwrap();
        assert!((low[RECORDS_T - 1]..=anchors[1]).contains(&lower));
        assert!((anchors[SPINE_K - 2]..=high[0]).contains(&upper));
    }

    #[test]
    fn mixture_sweep_is_monotone() {
        let old = std::array::from_fn(|j| probit(rank_grid()[j]));
        let batch = [-1.0, -0.5, 0.0, 1.0, 2.0];
        let targets = shifted_ranks(0.17);
        let mut output = [0.0; SPINE_K];
        blend_reanchor(
            &old,
            &batch,
            0.3,
            &targets,
            &targets.map(probit),
            SpineLink::Probit,
            &mut output,
        );
        assert!(output.windows(2).all(|window| window[0] <= window[1]));
    }
}

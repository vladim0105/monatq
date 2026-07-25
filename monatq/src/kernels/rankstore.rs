use rayon::prelude::*;
use std::mem::size_of;

use crate::{
    kernels::{DigestKernel, RankStore, RankStoreConfig, sealed},
    tensor_digest::StorageOperations,
};

pub const RANKSTORE_K: usize = 64;
const MASS_QUANTA: u64 = u16::MAX as u64;

/// The complete persistent summary for one tensor position.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RankStoreState {
    values: [f32; RANKSTORE_K],
    masses: [u16; RANKSTORE_K],
    pure_mask: u64,
    min: f32,
    max: f32,
}

impl Default for RankStoreState {
    fn default() -> Self {
        Self {
            values: [0.0; RANKSTORE_K],
            masses: [0; RANKSTORE_K],
            pure_mask: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }
}

const _: () = assert!(size_of::<RankStoreState>() == 400);

#[derive(Clone, Copy)]
struct Entry {
    value: f32,
    weight: u64,
    pure: bool,
}

pub(crate) struct RankStoreStorage {
    shape: Vec<usize>,
    numel: usize,
    config: RankStoreConfig,
    row_buffer: Vec<f32>,
    n_buffered: usize,
    sample_count: u64,
    states: Vec<RankStoreState>,
}

impl RankStoreStorage {
    pub(crate) fn with_config(shape: &[usize], config: RankStoreConfig) -> Self {
        assert!(
            config.buffer_capacity > 0,
            "buffer_capacity must be positive"
        );
        let numel = shape.iter().product::<usize>();
        let buffer_len = numel
            .checked_mul(config.buffer_capacity)
            .expect("row buffer size overflow");
        Self {
            shape: shape.to_vec(),
            numel,
            config,
            row_buffer: vec![0.0; buffer_len],
            n_buffered: 0,
            sample_count: 0,
            states: vec![RankStoreState::default(); numel],
        }
    }

    pub(crate) fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub(crate) fn config(&self) -> &RankStoreConfig {
        &self.config
    }

    pub(crate) fn min(&mut self) -> Vec<f32> {
        self.flush();
        self.states.iter().map(|state| state.min).collect()
    }

    pub(crate) fn max(&mut self) -> Vec<f32> {
        self.flush();
        self.states.iter().map(|state| state.max).collect()
    }

    pub(crate) fn cell_min(&mut self, idx: usize) -> f32 {
        self.flush();
        self.states[idx].min
    }

    pub(crate) fn cell_max(&mut self, idx: usize) -> f32 {
        self.flush();
        self.states[idx].max
    }

    pub(crate) fn state_memory_bytes(&self) -> usize {
        self.states.capacity() * size_of::<RankStoreState>()
    }

    pub(crate) const fn state_bytes_per_position(&self) -> usize {
        size_of::<RankStoreState>()
    }

    pub(crate) fn buffer_memory_bytes(&self) -> usize {
        self.row_buffer.capacity() * size_of::<f32>()
    }

    pub(crate) fn allocated_memory_bytes(&self) -> usize {
        self.shape.capacity() * size_of::<usize>()
            + self.state_memory_bytes()
            + self.buffer_memory_bytes()
    }
}

impl sealed::Kernel<f32> for RankStore {
    type Storage = RankStoreStorage;

    fn create_storage(shape: &[usize], config: RankStoreConfig) -> Self::Storage {
        RankStoreStorage::with_config(shape, config)
    }
}

impl DigestKernel<f32> for RankStore {
    type Config = RankStoreConfig;
}

impl StorageOperations<f32> for RankStoreStorage {
    fn numel(&self) -> usize {
        self.numel
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn total_weight(&self, idx: usize) -> u32 {
        assert!(idx < self.numel, "position index out of bounds");
        self.sample_count.min(u32::MAX as u64) as u32
    }

    fn update(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.numel, "tensor sample has the wrong length");
        assert!(
            !data.iter().any(|value| value.is_nan()),
            "RANKSTORE rejects tensor updates containing NaN"
        );
        if self.n_buffered == self.config.buffer_capacity {
            self.flush();
        }
        let start = self.n_buffered * self.numel;
        self.row_buffer[start..start + self.numel].copy_from_slice(data);
        self.n_buffered += 1;
        if self.n_buffered == self.config.buffer_capacity {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.n_buffered == 0 {
            return;
        }
        let rows = self.n_buffered;
        let numel = self.numel;
        let old_count = self.sample_count;
        let buffer = &self.row_buffer[..rows * numel];
        self.states.par_iter_mut().enumerate().for_each_init(
            || Vec::<Entry>::with_capacity(RANKSTORE_K + rows),
            |scratch, (position, state)| {
                scratch.clear();
                decode_old(state, old_count, scratch);
                for row in 0..rows {
                    scratch.push(Entry {
                        value: buffer[row * numel + position],
                        weight: MASS_QUANTA,
                        pure: true,
                    });
                }
                scratch.sort_by(|left, right| left.value.total_cmp(&right.value));
                let minimum = scratch.first().expect("nonempty flush").value;
                let maximum = scratch.last().expect("nonempty flush").value;
                coalesce(scratch);
                state.min = if old_count == 0 {
                    minimum
                } else {
                    min_value(state.min, minimum)
                };
                state.max = if old_count == 0 {
                    maximum
                } else {
                    max_value(state.max, maximum)
                };
                compress_and_store(scratch, state);
            },
        );
        self.sample_count = self.sample_count.saturating_add(rows as u64);
        self.n_buffered = 0;
    }

    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.flush();
        self.states.iter().map(|state| query(state, q)).collect()
    }

    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.flush();
        qs.iter()
            .map(|&q| self.states.iter().map(|state| query(state, q)).collect())
            .collect()
    }

    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        qs.iter().map(|&q| query(&self.states[idx], q)).collect()
    }
}

fn decode_old(state: &RankStoreState, old_count: u64, output: &mut Vec<Entry>) {
    if old_count == 0 {
        return;
    }
    for index in 0..RANKSTORE_K {
        let mass = state.masses[index];
        if mass == 0 {
            continue;
        }
        output.push(Entry {
            value: state.values[index],
            weight: u64::from(mass).saturating_mul(old_count),
            pure: state.pure_mask & (1_u64 << index) != 0,
        });
    }
}

fn coalesce(entries: &mut Vec<Entry>) {
    let mut write = 0;
    for read in 0..entries.len() {
        if write > 0 && entries[write - 1].value == entries[read].value {
            entries[write - 1].weight = entries[write - 1]
                .weight
                .saturating_add(entries[read].weight);
            entries[write - 1].pure &= entries[read].pure;
        } else {
            entries[write] = entries[read];
            write += 1;
        }
    }
    entries.truncate(write);
}

fn compress_and_store(entries: &[Entry], state: &mut RankStoreState) {
    let groups = groups(entries);
    let total = entries.iter().map(|entry| entry.weight).sum::<u64>();
    let mut representatives = Vec::with_capacity(groups.len());
    for &(start, end) in &groups {
        let mass = entries[start..end]
            .iter()
            .map(|entry| entry.weight)
            .sum::<u64>();
        let pure = end == start + 1 && entries[start].pure;
        let value = if pure || !entries[start].value.is_finite() {
            entries[start].value
        } else {
            let moment = entries[start..end]
                .iter()
                .map(|entry| entry.value as f64 * entry.weight as f64)
                .sum::<f64>();
            (moment / mass as f64) as f32
        };
        debug_assert!(value.is_finite() || pure);
        representatives.push(Entry {
            value,
            weight: mass,
            pure,
        });
    }
    // Adjacent weighted means can round to the same f32. Coalesce them so the
    // fixed knot budget is not wasted and a mixed representative never becomes pure.
    coalesce(&mut representatives);

    *state = RankStoreState {
        min: state.min,
        max: state.max,
        ..RankStoreState::default()
    };
    let mut cumulative = 0_u64;
    let mut previous_prefix = 0_u64;
    let mut output_index = 0;
    for representative in representatives {
        cumulative = cumulative.saturating_add(representative.weight);
        let prefix = round_ratio_ties_even(cumulative, MASS_QUANTA, total);
        let encoded = prefix - previous_prefix;
        previous_prefix = prefix;
        if encoded == 0 {
            continue;
        }
        state.values[output_index] = representative.value;
        state.masses[output_index] = encoded as u16;
        if representative.pure {
            state.pure_mask |= 1_u64 << output_index;
        }
        output_index += 1;
    }
}

fn groups(entries: &[Entry]) -> Vec<(usize, usize)> {
    if entries.len() <= RANKSTORE_K {
        return (0..entries.len()).map(|index| (index, index + 1)).collect();
    }
    let total = entries.iter().map(|entry| entry.weight).sum::<u64>();
    let mut prefix = Vec::with_capacity(entries.len() + 1);
    prefix.push(0_u64);
    for entry in entries {
        prefix.push(prefix.last().copied().unwrap().saturating_add(entry.weight));
    }
    let mut boundaries = Vec::with_capacity(RANKSTORE_K - 1);
    // Reserve cuts for infinities first: they are indivisible pure singleton groups and
    // must never enter a mean, even when an arcsine target does not land near them.
    if entries
        .first()
        .is_some_and(|entry| entry.value == f32::NEG_INFINITY)
    {
        boundaries.push(1);
    }
    if entries
        .last()
        .is_some_and(|entry| entry.value == f32::INFINITY)
    {
        boundaries.push(entries.len() - 1);
    }
    for cut in 1..RANKSTORE_K {
        let q = ((std::f64::consts::PI * cut as f64 / (2.0 * RANKSTORE_K as f64)).sin()).powi(2);
        let target = q * total as f64;
        let index = prefix[1..].partition_point(|&mass| (mass as f64) < target);
        let before = prefix[index] as f64;
        let after = prefix[index + 1] as f64;
        let boundary = if target - before <= after - target {
            index
        } else {
            index + 1
        };
        if boundary > 0
            && boundary < entries.len()
            && !boundaries.contains(&boundary)
            && boundaries.len() < RANKSTORE_K - 1
        {
            boundaries.push(boundary);
        }
    }
    boundaries.sort_unstable();
    let mut result = Vec::with_capacity(RANKSTORE_K);
    let mut start = 0;
    for boundary in boundaries {
        result.push((start, boundary));
        start = boundary;
    }
    result.push((start, entries.len()));

    while result.len() < RANKSTORE_K {
        let mut best: Option<(f64, usize, usize)> = None;
        for (group_index, &(left_index, right_index)) in result.iter().enumerate() {
            if right_index - left_index <= 1 {
                continue;
            }
            let left = arcsine_scale(prefix[left_index] as f64 / total as f64);
            let right = arcsine_scale(prefix[right_index] as f64 / total as f64);
            let midpoint = (left + right) * 0.5;
            let mut split = left_index + 1;
            let mut distance = f64::INFINITY;
            for (candidate, &candidate_prefix) in prefix
                .iter()
                .enumerate()
                .take(right_index)
                .skip(left_index + 1)
            {
                let candidate_distance =
                    (arcsine_scale(candidate_prefix as f64 / total as f64) - midpoint).abs();
                if candidate_distance < distance {
                    distance = candidate_distance;
                    split = candidate;
                }
            }
            let span = right - left;
            if best.is_none_or(|(best_span, best_index, _)| {
                span > best_span || (span == best_span && group_index < best_index)
            }) {
                best = Some((span, group_index, split));
            }
        }
        let Some((_, index, split)) = best else { break };
        let (start, end) = result[index];
        result[index] = (start, split);
        result.insert(index + 1, (split, end));
    }
    result
}

fn arcsine_scale(q: f64) -> f64 {
    (2.0 / std::f64::consts::PI) * q.clamp(0.0, 1.0).sqrt().asin()
}

fn round_ratio_ties_even(numerator: u64, multiplier: u64, denominator: u64) -> u64 {
    let scaled = (numerator as u128) * (multiplier as u128);
    let denominator = denominator as u128;
    let quotient = scaled / denominator;
    let remainder = scaled % denominator;
    let rounded =
        if remainder * 2 > denominator || (remainder * 2 == denominator && quotient & 1 == 1) {
            quotient + 1
        } else {
            quotient
        };
    rounded as u64
}

fn query(state: &RankStoreState, q: f32) -> f32 {
    if state.masses[0] == 0 {
        return 0.0;
    }
    if q.is_nan() {
        return f32::NAN;
    }
    if q <= 0.0 {
        return state.min;
    }
    if q >= 1.0 {
        return state.max;
    }
    let target = q as f64;
    let total = MASS_QUANTA as f64;
    let mut cumulative = 0_u64;
    for index in 0..RANKSTORE_K {
        let mass = u64::from(state.masses[index]);
        if mass == 0 {
            continue;
        }
        let left = cumulative as f64 / total;
        cumulative += mass;
        let right = cumulative as f64 / total;
        if state.pure_mask & (1_u64 << index) != 0 && target >= left && target <= right {
            return state.values[index];
        }
    }

    let first = state.masses.iter().position(|&mass| mass != 0).unwrap();
    let mut previous_rank = 0.0_f64;
    let mut previous_value = state.values[first];
    cumulative = 0;
    for index in 0..RANKSTORE_K {
        let mass = u64::from(state.masses[index]);
        if mass == 0 {
            continue;
        }
        let left = cumulative as f64 / total;
        cumulative += mass;
        let right = cumulative as f64 / total;
        let ranks = if state.pure_mask & (1_u64 << index) != 0 {
            [left, right]
        } else {
            [(left + right) * 0.5, f64::NAN]
        };
        for rank in ranks {
            if rank.is_nan() {
                continue;
            }
            let value = state.values[index];
            if target <= rank {
                return interpolate(previous_rank, previous_value, rank, value, target);
            }
            previous_rank = rank;
            previous_value = value;
        }
    }
    interpolate(
        previous_rank,
        previous_value,
        1.0,
        state.values[first_active_from_end(state)],
        target,
    )
}

fn first_active_from_end(state: &RankStoreState) -> usize {
    state.masses.iter().rposition(|&mass| mass != 0).unwrap()
}

fn interpolate(left_q: f64, left: f32, right_q: f64, right: f32, q: f64) -> f32 {
    if left == right || right_q <= left_q {
        return left;
    }
    if !left.is_finite() {
        return left;
    }
    if !right.is_finite() {
        return right;
    }
    let fraction = ((q - left_q) / (right_q - left_q)).clamp(0.0, 1.0);
    (left as f64 + (right as f64 - left as f64) * fraction) as f32
}

fn min_value(left: f32, right: f32) -> f32 {
    if left.total_cmp(&right).is_le() {
        left
    } else {
        right
    }
}

fn max_value(left: f32, right: f32) -> f32 {
    if left.total_cmp(&right).is_ge() {
        left
    } else {
        right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_state_is_exactly_400_bytes() {
        assert_eq!(size_of::<RankStoreState>(), 400);
        assert_eq!(std::mem::align_of::<RankStoreState>(), 8);
    }

    #[test]
    fn ties_even_integer_quantization() {
        assert_eq!(round_ratio_ties_even(1, 5, 2), 2);
        assert_eq!(round_ratio_ties_even(3, 5, 2), 8);
    }

    #[test]
    fn encoded_state_invariants_hold() {
        let mut storage = RankStoreStorage::with_config(
            &[1],
            RankStoreConfig {
                buffer_capacity: 17,
            },
        );
        for index in 0..2_003 {
            let value = match index {
                0 => f32::NEG_INFINITY,
                1 => f32::INFINITY,
                _ if index % 11 == 0 => 3.0,
                _ => ((index * 7919 % 1009) as f32 - 500.0) / 7.0,
            };
            storage.update(&[value]);
        }
        storage.flush();
        let state = &storage.states[0];
        assert_eq!(
            state
                .masses
                .iter()
                .map(|&mass| u64::from(mass))
                .sum::<u64>(),
            MASS_QUANTA
        );
        let active = (0..RANKSTORE_K)
            .filter(|&index| state.masses[index] != 0)
            .collect::<Vec<_>>();
        assert!(active.windows(2).all(|pair| {
            state.values[pair[0]]
                .total_cmp(&state.values[pair[1]])
                .is_le()
        }));
        for &index in &active {
            if !state.values[index].is_finite() {
                assert_ne!(state.pure_mask & (1_u64 << index), 0);
            }
        }
    }
}

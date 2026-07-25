use std::sync::OnceLock;

use rayon::prelude::*;

use crate::{
    kernels::{DigestKernel, RankKnot, RankKnotConfig, sealed},
    tensor_digest::StorageOperations,
};

pub const RANK_KNOT_K: usize = 32;
const MASS_QUANTA: u64 = u16::MAX as u64;

/// The complete persistent summary for one tensor position.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RankKnotState {
    values: [f32; RANK_KNOT_K],
    masses: [u16; RANK_KNOT_K],
    pure_mask: u64,
    min: f32,
    max: f32,
}

impl Default for RankKnotState {
    fn default() -> Self {
        Self {
            values: [0.0; RANK_KNOT_K],
            masses: [0; RANK_KNOT_K],
            pure_mask: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    value: f32,
    weight: u64,
    pure: bool,
}

struct RankKnotScratch {
    incoming: Vec<f32>,
    entries: Vec<Entry>,
    boundaries: [usize; RANK_KNOT_K - 1],
}

impl RankKnotScratch {
    fn new(rows: usize) -> Self {
        Self {
            incoming: Vec::with_capacity(rows),
            entries: Vec::with_capacity(RANK_KNOT_K + rows),
            boundaries: [0; RANK_KNOT_K - 1],
        }
    }
}

pub(crate) struct RankKnotStorage {
    shape: Vec<usize>,
    numel: usize,
    config: RankKnotConfig,
    row_buffer: Vec<f32>,
    n_buffered: usize,
    sample_count: u64,
    states: Vec<RankKnotState>,
}

impl RankKnotStorage {
    pub(crate) fn with_config(shape: &[usize], config: RankKnotConfig) -> Self {
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
            states: vec![RankKnotState::default(); numel],
        }
    }

    pub(crate) fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub(crate) fn config(&self) -> &RankKnotConfig {
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
}

impl sealed::Kernel<f32> for RankKnot {
    type Storage = RankKnotStorage;

    fn create_storage(shape: &[usize], config: RankKnotConfig) -> Self::Storage {
        RankKnotStorage::with_config(shape, config)
    }
}

impl DigestKernel<f32> for RankKnot {
    type Config = RankKnotConfig;
}

impl StorageOperations<f32> for RankKnotStorage {
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
        self.states
            .par_iter_mut()
            .with_min_len(64)
            .enumerate()
            .for_each_init(
                || RankKnotScratch::new(rows),
                |scratch, (position, state)| {
                    scratch.incoming.clear();
                    scratch
                        .incoming
                        .extend(buffer.chunks_exact(numel).map(|row| row[position]));
                    scratch
                        .incoming
                        .sort_unstable_by(|left, right| left.partial_cmp(right).unwrap());
                    let minimum = scratch.incoming[0];
                    let maximum = scratch.incoming[rows - 1];

                    merge_old_and_incoming(
                        state,
                        old_count,
                        &scratch.incoming,
                        &mut scratch.entries,
                    );
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
                    compress_and_store(&scratch.entries, state, &mut scratch.boundaries);
                },
            );
        self.sample_count = self.sample_count.saturating_add(rows as u64);
        self.n_buffered = 0;
    }

    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.flush();
        self.states
            .par_iter()
            .map(|state| query(state, q))
            .collect()
    }

    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.flush();
        qs.iter()
            .map(|&q| {
                self.states
                    .par_iter()
                    .map(|state| query(state, q))
                    .collect()
            })
            .collect()
    }

    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.flush();
        assert!(idx < self.numel, "position index out of bounds");
        qs.iter().map(|&q| query(&self.states[idx], q)).collect()
    }
}

fn merge_old_and_incoming(
    state: &RankKnotState,
    old_count: u64,
    incoming: &[f32],
    output: &mut Vec<Entry>,
) {
    output.clear();
    if old_count == 0 {
        for &value in incoming {
            push_coalesced(
                output,
                Entry {
                    value,
                    weight: MASS_QUANTA,
                    pure: true,
                },
            );
        }
        return;
    }
    let mut old_index = 0;
    let mut new_index = 0;
    while old_index < RANK_KNOT_K || new_index < incoming.len() {
        while old_index < RANK_KNOT_K && state.masses[old_index] == 0 {
            old_index += 1;
        }
        let take_old = old_index < RANK_KNOT_K
            && (new_index == incoming.len()
                || state.values[old_index]
                    .total_cmp(&incoming[new_index])
                    .is_le());
        if take_old {
            push_coalesced(
                output,
                Entry {
                    value: state.values[old_index],
                    weight: u64::from(state.masses[old_index]).saturating_mul(old_count),
                    pure: state.pure_mask & (1_u64 << old_index) != 0,
                },
            );
            old_index += 1;
        } else if new_index < incoming.len() {
            push_coalesced(
                output,
                Entry {
                    value: incoming[new_index],
                    weight: MASS_QUANTA,
                    pure: true,
                },
            );
            new_index += 1;
        } else {
            break;
        }
    }
}

fn push_coalesced(entries: &mut Vec<Entry>, entry: Entry) {
    if let Some(last) = entries.last_mut()
        && last.value == entry.value
    {
        last.weight = last.weight.saturating_add(entry.weight);
        last.pure &= entry.pure;
    } else {
        entries.push(entry);
    }
}

fn compress_and_store(
    entries: &[Entry],
    state: &mut RankKnotState,
    boundaries: &mut [usize; RANK_KNOT_K - 1],
) {
    let (total, boundary_count) = make_boundaries(entries, boundaries);
    *state = RankKnotState {
        min: state.min,
        max: state.max,
        ..RankKnotState::default()
    };

    let mut start = 0;
    let mut pending: Option<Entry> = None;
    let mut cumulative = 0_u64;
    let mut previous_prefix = 0_u64;
    let mut output_index = 0;
    for end in boundaries[..boundary_count]
        .iter()
        .copied()
        .chain(std::iter::once(entries.len()))
    {
        let pure = end == start + 1 && entries[start].pure;
        let (mass, value) = if pure {
            (entries[start].weight, entries[start].value)
        } else {
            let (mass, moment) =
                entries[start..end]
                    .iter()
                    .fold((0_u64, 0.0_f64), |(mass, moment), entry| {
                        (
                            mass.saturating_add(entry.weight),
                            moment + entry.value as f64 * entry.weight as f64,
                        )
                    });
            let value = if entries[start].value.is_finite() {
                (moment / mass as f64) as f32
            } else {
                entries[start].value
            };
            (mass, value)
        };
        debug_assert!(value.is_finite() || pure);
        let representative = Entry {
            value,
            weight: mass,
            pure,
        };
        if let Some(previous) = pending.as_mut()
            && previous.value == representative.value
        {
            previous.weight = previous.weight.saturating_add(representative.weight);
            previous.pure &= representative.pure;
        } else if let Some(previous) = pending.replace(representative) {
            store_representative(
                state,
                previous,
                total,
                &mut cumulative,
                &mut previous_prefix,
                &mut output_index,
            );
        }
        start = end;
    }
    if let Some(previous) = pending {
        store_representative(
            state,
            previous,
            total,
            &mut cumulative,
            &mut previous_prefix,
            &mut output_index,
        );
    }
}

fn store_representative(
    state: &mut RankKnotState,
    representative: Entry,
    total: u64,
    cumulative: &mut u64,
    previous_prefix: &mut u64,
    output_index: &mut usize,
) {
    *cumulative = cumulative.saturating_add(representative.weight);
    let prefix = round_div_ties_even(*cumulative, total / MASS_QUANTA);
    let encoded = prefix - *previous_prefix;
    *previous_prefix = prefix;
    if encoded == 0 {
        return;
    }
    state.values[*output_index] = representative.value;
    state.masses[*output_index] = encoded as u16;
    if representative.pure {
        state.pure_mask |= 1_u64 << *output_index;
    }
    *output_index += 1;
}

fn make_boundaries(entries: &[Entry], boundaries: &mut [usize; RANK_KNOT_K - 1]) -> (u64, usize) {
    let total = entries.iter().map(|entry| entry.weight).sum();
    if entries.len() <= RANK_KNOT_K {
        for (slot, boundary) in boundaries.iter_mut().zip(1..entries.len()) {
            *slot = boundary;
        }
        return (total, entries.len() - 1);
    }
    let mut boundary_count = 0;
    // Infinities are indivisible singleton groups and must never enter a mean.
    // Targets are ordered, so boundaries can be emitted in order without sorting.
    let positive_infinity = entries
        .last()
        .is_some_and(|entry| entry.value == f32::INFINITY);
    if entries
        .first()
        .is_some_and(|entry| entry.value == f32::NEG_INFINITY)
    {
        boundaries[boundary_count] = 1;
        boundary_count += 1;
    }
    let target_limit = RANK_KNOT_K - 1 - usize::from(positive_infinity);
    let mut index = 0;
    let mut before = 0_u64;
    let mut after = entries[0].weight;
    for &q in arcsine_targets() {
        let target = q * total as f64;
        while index + 1 < entries.len() && (after as f64) < target {
            before = after;
            index += 1;
            after = after.saturating_add(entries[index].weight);
        }
        let before = before as f64;
        let after = after as f64;
        let boundary = if target - before <= after - target {
            index
        } else {
            index + 1
        };
        if boundary > 0
            && boundary < entries.len()
            && (boundary_count == 0 || boundaries[boundary_count - 1] != boundary)
            && boundary_count < target_limit
        {
            boundaries[boundary_count] = boundary;
            boundary_count += 1;
        }
    }
    if positive_infinity
        && (boundary_count == 0 || boundaries[boundary_count - 1] != entries.len() - 1)
    {
        boundaries[boundary_count] = entries.len() - 1;
        boundary_count += 1;
    }
    (total, boundary_count)
}

fn arcsine_targets() -> &'static [f64; RANK_KNOT_K - 1] {
    static TARGETS: OnceLock<[f64; RANK_KNOT_K - 1]> = OnceLock::new();
    TARGETS.get_or_init(|| {
        std::array::from_fn(|index| {
            let cut = index + 1;
            (std::f64::consts::PI * cut as f64 / (2.0 * RANK_KNOT_K as f64))
                .sin()
                .powi(2)
        })
    })
}

fn round_div_ties_even(numerator: u64, denominator: u64) -> u64 {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let half = denominator / 2;
    if remainder > half || (denominator & 1 == 0 && remainder == half && quotient & 1 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

fn query(state: &RankKnotState, q: f32) -> f32 {
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
    let target = q as f64 * MASS_QUANTA as f64;
    let first = state.masses.iter().position(|&mass| mass != 0).unwrap();
    let mut previous_rank = 0.0_f64;
    let mut previous_value = state.values[first];
    let mut cumulative = 0_u64;
    for index in 0..RANK_KNOT_K {
        let mass = u64::from(state.masses[index]);
        if mass == 0 {
            continue;
        }
        let left = cumulative as f64;
        cumulative += mass;
        let right = cumulative as f64;
        let value = state.values[index];
        if state.pure_mask & (1_u64 << index) != 0 {
            if target >= left && target <= right {
                return value;
            }
            if target <= left {
                return interpolate(previous_rank, previous_value, left, value, target);
            }
            previous_rank = right;
            previous_value = value;
        } else {
            let rank = (left + right) * 0.5;
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
        MASS_QUANTA as f64,
        state.values[first_active_from_end(state)],
        target,
    )
}

fn first_active_from_end(state: &RankKnotState) -> usize {
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
    fn ties_even_integer_quantization() {
        assert_eq!(round_div_ties_even(5, 2), 2);
        assert_eq!(round_div_ties_even(15, 2), 8);
    }

    #[test]
    fn encoded_state_invariants_hold() {
        let mut storage = RankKnotStorage::with_config(
            &[1],
            RankKnotConfig {
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
        let active = (0..RANK_KNOT_K)
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

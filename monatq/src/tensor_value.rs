/// Trait for types that can be stored in a `TensorDigest`.
///
/// T-Digest centroids and weights always stay `f32` (algorithm requirement).
/// This trait provides the conversion needed when incoming values are consumed
/// during `compress()`.
pub trait TensorValue: Copy + Send + Sync + 'static + PartialOrd {
    /// One-byte tag stored as the first field of every saved `TensorDigest` file.
    const DTYPE_TAG: u8;
    /// Convert this value to `f32` for use in centroid arithmetic.
    fn to_f32(self) -> f32;
    /// Construct a value from its `f32` representation (used for centroid-derived bounds).
    fn from_f32(f: f32) -> Self;
    /// Returns `true` if this value is considered non-zero.
    fn is_nonzero(self) -> bool;
    /// Sentinel used to initialise minimum trackers (should be the type's maximum value).
    fn min_sentinel() -> Self;
    /// Sentinel used to initialise maximum trackers (should be the type's minimum value).
    fn max_sentinel() -> Self;
}

impl TensorValue for f32 {
    const DTYPE_TAG: u8 = 0;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(f: f32) -> Self {
        f
    }
    #[inline(always)]
    fn is_nonzero(self) -> bool {
        self.abs() > 1e-12
    }
    fn min_sentinel() -> Self {
        f32::INFINITY
    }
    fn max_sentinel() -> Self {
        f32::NEG_INFINITY
    }
}

impl TensorValue for i32 {
    const DTYPE_TAG: u8 = 1;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn from_f32(f: f32) -> Self {
        f as i32
    }
    #[inline(always)]
    fn is_nonzero(self) -> bool {
        self != 0
    }
    fn min_sentinel() -> Self {
        i32::MAX
    }
    fn max_sentinel() -> Self {
        i32::MIN
    }
}

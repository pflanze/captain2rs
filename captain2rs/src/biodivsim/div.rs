use std::ops::{Mul, Range};

pub type Float = f64;

#[inline]
pub fn square<Number: Mul + Copy>(x: Number) -> Number::Output {
    x * x
}

/// The range of integers before and after `i` by the amount of
/// `threshold` (exclusive the latter bound), limited to the range
/// `0..length`.
pub(crate) fn bounded_range_around(i: u32, length: u32, threshold: u32) -> Range<u32> {
    let n_min = i.saturating_sub(threshold);
    let n_max = u32::min(length, i + threshold);
    n_min..n_max
}

/// Same as `bounded_range_around` but does not include the lower
/// bound, and instead includes the upper bound; also returns whether
/// the threshold window is unclipped. `threshold` must be >= 1.
pub(crate) fn flipped_bounded_range_around(
    i: usize,
    length: usize,
    threshold: usize,
) -> (Range<usize>, bool) {
    let n_min = i.saturating_sub(threshold - 1);
    let n_max = usize::min(length, i + threshold + 1);
    (
        n_min..n_max,
        i >= threshold && (i + threshold + 1) <= length,
    )
}

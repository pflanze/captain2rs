use std::ops::{Mul, Range};

pub type Float = f64;

#[inline]
pub fn square<Number: Mul + Copy>(x: Number) -> Number::Output {
    x * x
}

/// Same as `bounded_range_around` but does not include the lower
/// bound, and instead includes the upper bound; also returns whether
/// the threshold window is unclipped. `threshold` must be >= 1.
pub fn flipped_bounded_range_around(
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

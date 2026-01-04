use std::ops::{Mul, Range};

use noisy_float::types::R64;

pub type Float = f64;
pub type RealFloat = R64;

const fn enforce_realfloat_to_float_relation() {
    const {
        assert!(size_of::<Float>() == size_of::<RealFloat>());
    }
}

#[inline]
pub fn square<Number: Mul + Copy>(x: Number) -> Number::Output {
    x * x
}

/// The range of integers before and after `i` by the amount of
/// `threshold`, limited to the range `0..length`.
pub(crate) fn bounded_range_around(i: u32, length: u32, threshold: u32) -> Range<u32> {
    let n_min = i.saturating_sub(threshold);
    let n_max = u32::min(length, i + threshold + 1);
    n_min..n_max
}

/// Same as `bounded_range_around` but also returns whether the
/// threshold window is unclipped. `threshold` must be >= 1.
pub(crate) fn bounded_range_around_w_clipped(
    i: usize,
    length: usize,
    threshold: usize,
) -> (Range<usize>, bool) {
    let n_min = i.saturating_sub(threshold);
    let n_max = usize::min(length, i + threshold + 1);
    (
        n_min..n_max,
        i >= threshold && (i + threshold + 1) <= length,
    )
}

#[test]
fn t_bounded_range_around_w_clipped() {
    let t = |i, length, threshold| {
        let (range, unclipped) = bounded_range_around_w_clipped(i, length, threshold);
        let range0 = bounded_range_around(i as u32, length as u32, threshold as u32);
        let range0 = range0.start as usize..range0.end as usize;
        assert_eq!(range, range0);
        (range, unclipped)
    };
    assert_eq!(t(0, 5, 1), (0..2, false)); // [ p, x, _, _, _ ]
    assert_eq!(t(1, 5, 1), (0..3, true)); //  [ x, p, x, _, _ ]
    assert_eq!(t(2, 5, 1), (1..4, true)); //  [ _, x, p, x, _ ]
    assert_eq!(t(3, 5, 1), (2..5, true)); //  [ _, _, x, p, x ]
    assert_eq!(t(4, 5, 1), (3..5, false)); // [ _, _, _, x, p ]

    assert_eq!(t(0, 6, 2), (0..3, false)); // [ p, x, x, _, _, _ ]
    assert_eq!(t(1, 6, 2), (0..4, false)); // [ x, p, x, x, _, _ ]
    assert_eq!(t(2, 6, 2), (0..5, true)); //  [ x, x, p, x, x, _ ]
    assert_eq!(t(3, 6, 2), (1..6, true)); //  [ _, x, x, p, x, x ]
    assert_eq!(t(4, 6, 2), (2..6, false)); // [ _, _, x, x, p, x ]
    assert_eq!(t(5, 6, 2), (3..6, false)); // [ _, _, _, x, x, p ]
}

#[derive(Debug)]
pub struct FlippedBoundedRangesAround {
    pub n: usize,
    pub m: usize,
    pub n_range: Range<usize>,
    pub n_is_unclipped: bool,
    pub m_range: Range<usize>,
    pub m_is_unclipped: bool,
}

impl FlippedBoundedRangesAround {
    #[inline]
    pub fn new(n: usize, dim_n: usize, m: usize, dim_m: usize, threshold: usize) -> Self {
        let (n_range, n_is_unclipped) = bounded_range_around_w_clipped(n, dim_n, threshold);
        let (m_range, m_is_unclipped) = bounded_range_around_w_clipped(m, dim_m, threshold);
        FlippedBoundedRangesAround {
            n,
            m,
            n_range,
            n_is_unclipped,
            m_range,
            m_is_unclipped,
        }
    }
}

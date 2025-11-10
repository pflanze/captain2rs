use std::mem::MaybeUninit;
use std::ops::{Mul, Range};

use crate::coo::Coo;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, Axis};
use numpy::{
    pyo3::{prelude::*, pyclass},
    PyArray2, PyArray3, PyReadonlyArray,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

type Float = f64;

fn square<Number: Mul + Copy>(x: Number) -> Number::Output {
    x * x
}

/// The range of integers before and after `i` by the amount of
/// `threshold` (exclusive the latter bound), limited to the range
/// `0..length`.
fn bounded_range_around(i: u32, length: u32, threshold: u32) -> Range<u32> {
    let n_min = i.saturating_sub(threshold);
    let n_max = u32::min(length, i + threshold);
    n_min..n_max
}

/// Same as `bounded_range_around` but does not include the lower
/// bound, and instead includes the upper bound; also returns whether
/// the threshold window is unclipped. `threshold` must be >= 1.
fn flipped_bounded_range_around(i: u32, length: u32, threshold: u32) -> (Range<u32>, bool) {
    let n_min = i.saturating_sub(threshold - 1);
    let n_max = u32::min(length, i + threshold + 1);
    (
        n_min..n_max,
        i >= threshold && (i + threshold + 1) <= length,
    )
}

// @jit(nopython=True)
// def dispersalDistancesThreshold(length: int,
//                                 lambda_0: float,
//                                 threshold=3):
//     print("calculating distances with threshold...")
//     dumping_dist = np.zeros((length, length, length, length))
//     for i in range(0, length):
//         for j in range(0, length):
//             for n in range(max([0, i-threshold]), min([length, i+threshold])):
//                 for m in range(max([0, j-threshold]), min([length, j+threshold])):
//                     exp_rate = 1.0 / lambda_0
//                     # relative dispersal probability: always 1 at distance = 0
//                     # the actual number of offspring is modulated by growth_rate
//                     dumping_dist[i, j, n, m] = np.exp(
//                         -exp_rate * np.sqrt((i - n) ** 2 + (j - m) ** 2)
//                     )
//     return dumping_dist

/// Compute dispersal distances with a threshold.
///
/// # Arguments
/// * `length` - size of the grid (length x length)
/// * `lambda_0` - dispersal parameter
/// * `threshold` - neighborhood radius
///
/// # Returns
/// A 4D array of shape (length, length, length, length)
pub fn dispersal_distances_threshold(
    length: u32,
    lambda_0: Float,
    threshold: u32,
    test_hack: bool,
    default: Option<Float>,
) -> Coo<i64, 4, Float> {
    println!("calculating distances with threshold...");

    let mut dumping_dist = Coo::new(default.unwrap_or(0.));
    let exp_rate = 1.0 / lambda_0;

    for i in 0..length {
        for j in 0..length {
            for n in bounded_range_around(i, length, threshold) {
                for m in bounded_range_around(j, length, threshold) {
                    let dx = (i as Float - n as Float).powi(2); // XX why not in i64!
                    let dy = (j as Float - m as Float).powi(2);
                    let dist = (dx + dy).sqrt();
                    dumping_dist
                        .insert([i, j, n, m], (-exp_rate * dist).exp())
                        .expect("inserts to happen in sorted order");
                }
            }
        }
    }

    if test_hack {
        dumping_dist.insert_unordered([5, 9, 7, 1], -7.3456);
    }

    dumping_dist
}

/// Reimplementation of `dispersal_distances_threshold`, calculating
/// only a single threshold area and re-using it for all points in
/// lookups instead of calculating a sparse array with an area for all
/// input points.
#[pyclass(frozen)]
#[derive(Debug)]
pub struct DispersalDistancesThreshold {
    pub threshold: u32,
    pub neg_exp_rate: Float,
    pub cache: Array2<Float>,
}

fn dist_value(neg_exp_rate: Float, dx: i32, dy: i32) -> Float {
    (neg_exp_rate * Float::sqrt((square(dx) + square(dy)) as Float)).exp()
}

#[pymethods]
impl DispersalDistancesThreshold {
    #[new]
    fn new(lambda_0: Float, threshold: u32) -> Self {
        let neg_exp_rate = -1.0 / lambda_0;
        let length = threshold * 2;
        let mut cache = Array2::zeros((length as usize, length as usize));
        for i in 0..length {
            for j in 0..length {
                cache[(i as usize, j as usize)] = dist_value(
                    neg_exp_rate,
                    i as i32 - threshold as i32,
                    j as i32 - threshold as i32,
                );
            }
        }

        Self {
            threshold,
            neg_exp_rate,
            cache,
        }
    }

    fn precise_at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        dist_value(self.neg_exp_rate, i as i32 - n as i32, j as i32 - m as i32)
    }

    /// Only valid in range of differences as per `self.threshold`,
    /// panics otherwise
    #[inline]
    fn cached_at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        self.cache[(
            (self.threshold + i - n) as usize,
            (self.threshold + j - m) as usize,
        )]
    }

    /// Run both `cached_at` and `precise_at` and assert equivalence
    /// for testing.
    fn asserting_at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        let val1 = self.precise_at(i, j, n, m);
        let val2 = self.cached_at(i, j, n, m);
        assert_eq!(val1, val2);
        val2
    }

    /// Only valid in range of differences as per `self.threshold`,
    /// may panic otherwise
    #[inline]
    fn at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        self.cached_at(i, j, n, m)
    }

    // (Unidiomatically, this has to panic for python errors, because
    // ndarray does not have a `try_map`. But pyo3 converts panics to
    // exceptions, so it ends up being ~the same as long as panics are
    // not overridden.)
    #[pyo3(name = "map")]
    fn map_rs<'py>(&self, py: Python<'py>, dot_transform: Py<PyAny>) -> Self {
        self.map(|x| {
            dot_transform
                .call(py, (x,), None)
                .expect("map() expects a Python callable with one argument")
                .extract(py)
                .expect("map(): callable did not return a floating point value")
        })
    }

    /// Calculate the equivalent of `einsum("ij,ijnm->nm", a, self)`
    #[pyo3(name = "apply")]
    fn apply_rs<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 2]>>,
    ) -> Bound<'py, PyArray2<Float>> {
        PyArray2::from_owned_array(py, self.apply(a.as_array()))
    }

    fn print(&self) {
        println!("{self:#?}");
    }
}

impl DispersalDistancesThreshold {
    /// Used internally. `a.is_standard_layout()` must be true!
    #[inline]
    fn dot<const M: u32>(
        &self,
        (n_range, m_start, a): (Range<u32>, u32, &ArrayView2<Float>),
    ) -> Float {
        let mut sum = 0.;
        let n_range_start = n_range.start;
        for n in n_range {
            let a_row = a.row(n as usize);
            let a_slice = &a_row.as_slice().unwrap()[m_start as usize..];
            assert!(a_slice.len() >= M as usize);
            let cache_row = self.cache.row((n - n_range_start) as usize);
            let cache_slice = &cache_row.as_slice().unwrap();
            assert!(cache_slice.len() >= M as usize);
            for _m in 0..M {
                sum += a_slice[_m as usize] * cache_slice[_m as usize];
            }
        }
        sum
    }

    fn map(&self, dot_transform: impl Fn(Float) -> Float) -> Self {
        let Self {
            threshold,
            neg_exp_rate,
            cache,
        } = self;
        let cache = cache.map(|x| dot_transform(*x));
        Self {
            threshold: *threshold,
            neg_exp_rate: *neg_exp_rate,
            cache,
        }
    }

    /// Calculate the equivalent of `einsum("ij,ijnm->nm", a, self)`
    /// and write the result to `c`. Initializes/overwrites all values
    /// in `c`
    fn apply_to(&self, a: ArrayView2<Float>, mut c: ArrayViewMut2<MaybeUninit<Float>>) {
        let DispersalDistancesThreshold {
            threshold,
            neg_exp_rate: _,
            cache: _,
        } = *self;

        let shape = a.dim();

        // Logically, apply the following (where b is the sparse array):

        // for p in 0..shape.0 {
        //     for q in 0..shape.1 {
        //         nm += a[(p, q)] * b.at(p, q);
        //     }
        // }

        let dim_i = shape.0 as u32;
        let dim_j = shape.1 as u32;
        assert!(a.is_standard_layout());
        assert!(self.cache.is_standard_layout());

        for i in 0..dim_i {
            for j in 0..dim_j {
                let (n_range, n_is_unclipped) = flipped_bounded_range_around(i, dim_i, threshold);
                let (m_range, m_is_unclipped) = flipped_bounded_range_around(j, dim_j, threshold);

                // The calculation we apply, logically, and literally
                // if there's no optimization:
                let fallback = || {
                    let mut sum = 0.;
                    for n in n_range.clone() {
                        for m in m_range.clone() {
                            sum += a[(n as usize, m as usize)] * self.at(i, j, n, m)
                        }
                    }
                    sum
                };
                // If the threshold area is not clipped by the image
                // boundaries, instantiate the `self.mult_and_sum`
                // method for a number of threshold area widths (to
                // make the compiler use SIMD instructions) and use
                // the corresponding version if we have one; otherwise
                // use the fallback. You can add more if you need a
                // `threshold` larger than 10 (the range len is twice
                // the threshold).
                let sum = if n_is_unclipped && m_is_unclipped {
                    let args = (n_range.clone(), m_range.start, &a);
                    match m_range.len() {
                        2 => self.dot::<2>(args),
                        4 => self.dot::<4>(args),
                        6 => self.dot::<6>(args),
                        8 => self.dot::<8>(args),
                        10 => self.dot::<10>(args),
                        12 => self.dot::<12>(args),
                        14 => self.dot::<14>(args),
                        16 => self.dot::<16>(args),
                        18 => self.dot::<18>(args),
                        20 => self.dot::<20>(args),
                        _ => fallback(),
                    }
                } else {
                    fallback()
                };
                c[(i as usize, j as usize)].write(sum);
            }
        }
    }

    fn apply(&self, a: ArrayView2<Float>) -> Array2<Float> {
        let mut c = Array2::<Float>::uninit(a.dim());
        self.apply_to(a, c.view_mut());
        // Safe because `.apply_to` overwrites all values in `c`
        unsafe { c.assume_init() }
    }
}

/// Equivalent of
///
///     NumCandidates = np.array(
///           [sparse.einsum("ij,ijnm->nm", self._h[i],
///                          self._dumping_dist ** (1 / self._lambda_0[i])
///                    ).todense() for i in range(self._n_species)])
///
/// if `self._dumping_dist` is coming from
///
///      dispersal_distances_threshold_rs(
///          self._h[i].shape[0],
///          lambda_0_init,
///          threshold)
///
#[pyfunction]
fn num_candidates_rs<'py>(
    py: Python<'py>,
    n_species: usize,
    lambda_0_init: Float,
    threshold: u32,
    lambda_0: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 1]>>,
    h: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 3]>>,
) -> Bound<'py, PyArray3<Float>> {
    let lambda_0 = lambda_0.as_array();
    let h = h.as_array();

    {
        let n = lambda_0.dim();
        if n < n_species {
            panic!("n_species {n_species} is higher than the number of values {n} in `lambda_0`")
        }
    }

    let (n, o, p) = h.dim();
    if n < n_species {
        panic!("n_species {n_species} is higher than the number of subarrays {n} in `h`")
    }

    let mut result = Array3::<Float>::uninit((n_species, o, p));
    let ddt = DispersalDistancesThreshold::new(lambda_0_init, threshold);
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, res)| {
            ddt.map(|x| x.powf(1. / lambda_0[i]))
                .apply_to(h.index_axis(Axis(0), i), res);
        });

    // Safe because we iterate over all of axis 0, then overwrite all
    // values in the iterated-over matrices via `.apply_to`, leaving
    // nothing uninitialized.
    PyArray3::from_owned_array(py, unsafe { result.assume_init() })
}

/// Compute dispersal distances using geographic coordinates with a threshold.
///
/// # Arguments
/// * `length` - size of the grid (length x length)
/// * `lambda_0` - dispersal parameter
/// * `lat` - 2D array of latitude coordinates (length x length)
/// * `lon` - 2D array of longitude coordinates (length x length)
/// * `threshold` - neighborhood radius (coordinate units)
///
/// # Returns
/// A 4D Coo sparse array of shape (length, length, length, length)
fn dispersal_distances_coord(
    length: u32,
    lambda_0: Float,
    lat: ArrayView2<Float>,
    lon: ArrayView2<Float>,
    threshold: Float,
) -> Coo<i64, 4, Float> {
    assert_eq!(lat.shape(), [length as usize, length as usize]);
    assert_eq!(lon.shape(), [length as usize, length as usize]);

    println!("calculating distances with coordinate threshold...");

    let mut dumping_dist = Coo::new(0.0);
    let exp_rate = 1.0 / lambda_0;

    let len_usize = length as usize;

    for i in 0..len_usize {
        for j in 0..len_usize {
            let lat_ij = lat[(i, j)];
            let lon_ij = lon[(i, j)];

            for n in 0..len_usize {
                for m in 0..len_usize {
                    let lat_nm = lat[(n, m)];
                    let lon_nm = lon[(n, m)];

                    let lat_diff = (lat_ij - lat_nm).abs();
                    let lon_diff = (lon_ij - lon_nm).abs();

                    if lat_diff <= threshold && lon_diff <= threshold {
                        // Calculate Euclidean distance in coordinate space
                        let dx = (lat_ij - lat_nm).powi(2);
                        let dy = (lon_ij - lon_nm).powi(2);
                        let dist = (dx + dy).sqrt();

                        // Calculate dispersal probability and insert into the sparse matrix
                        dumping_dist
                            .insert(
                                [i as i64, j as i64, n as i64, m as i64],
                                (-exp_rate * dist).exp(),
                            )
                            .expect("inserts to happen in sorted order");
                    }
                }
            }
        }
    }

    dumping_dist
}

// --- PyO3 wrapper functions and exports ---
#[pymodule]
mod captain2rs {
    use numpy::{pyo3::prelude::*, PyReadonlyArray};

    use super::{dispersal_distances_coord, dispersal_distances_threshold, Float};

    #[pyfunction]
    pub fn dispersal_distances_coord_rs<'py>(
        py: Python<'py>,
        length: u32,
        lambda_0: Float,
        lat: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 2]>>,
        lon: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 2]>>,
        threshold: Float,
    ) -> PyResult<Bound<'py, PyAny>> {
        dispersal_distances_coord(length, lambda_0, lat.as_array(), lon.as_array(), threshold)
            .to_python_sparse(py)
    }

    #[pyfunction]
    pub fn dispersal_distances_threshold_rs<'py>(
        py: Python<'py>,
        length: u32,
        lambda_0: Float,
        threshold: u32,
    ) -> pyo3::PyResult<Bound<'py, PyAny>> {
        dispersal_distances_threshold(length, lambda_0, threshold, false, None).to_python_sparse(py)
    }

    #[pyfunction]
    pub fn dispersal_distances_threshold_test_rs<'py>(
        py: Python<'py>,
        length: u32,
        lambda_0: Float,
        threshold: u32,
        test_hack: bool,
        default: Option<Float>,
    ) -> pyo3::PyResult<Bound<'py, PyAny>> {
        dispersal_distances_threshold(length, lambda_0, threshold, test_hack, default)
            .to_python_sparse(py)
    }

    #[pymodule_export]
    use super::num_candidates_rs;
    #[pymodule_export]
    use super::DispersalDistancesThreshold;
}

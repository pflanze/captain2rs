use std::mem::MaybeUninit;
use std::ops::Range;

use crate::coo::Coo;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, Axis};
use numpy::{
    pyo3::{self, prelude::*},
    PyArray2, PyArray3, PyReadonlyArray,
};
use pyo3::pyclass;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

type Float = f64;

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

fn square_i32(x: i32) -> i32 {
    x * x
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

#[pyclass(frozen)]
#[derive(Debug)]
pub struct DispersalDistancesThreshold {
    pub threshold: u32,
    pub neg_exp_rate: Float,
    pub cache: Array2<Float>,
}

fn dist_value(neg_exp_rate: Float, dx: i32, dy: i32) -> Float {
    (neg_exp_rate * Float::sqrt((square_i32(dx) + square_i32(dy)) as Float)).exp()
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

    fn asserting_at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        let val1 = self.precise_at(i, j, n, m);
        let val2 = self.cached_at(i, j, n, m);
        assert_eq!(val1, val2);
        val2
    }

    #[inline]
    fn at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        self.cached_at(i, j, n, m)
    }

    // (Ugly: panics for python errors, because ndarray does not have
    // a `try_map`. But pyo3 converts panics to exceptions, so no big
    // deal.)
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
    /// `a.is_standard_layout()` must be true!
    #[inline]
    fn mult_and_sum<const M: u32>(
        &self,
        n_range: Range<u32>,
        m_start: u32,
        a: &ArrayView2<Float>,
    ) -> Float {
        let mut sum = 0.;
        let n_range_start = n_range.start;
        for n in n_range {
            let a_row = a.row(n as usize);
            let a_ptr = &a_row.as_slice().unwrap()[m_start as usize..];
            assert!(a_ptr.len() >= M as usize);
            let cache_row = self.cache.row((n - n_range_start) as usize);
            let cache_ptr = &cache_row.as_slice().unwrap();
            assert!(cache_ptr.len() >= M as usize);
            for _m in 0..M {
                sum += a_ptr[_m as usize] * cache_ptr[_m as usize];
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

    fn apply_to(&self, a: ArrayView2<Float>, mut c: ArrayViewMut2<MaybeUninit<Float>>) {
        let DispersalDistancesThreshold {
            threshold,
            neg_exp_rate: _,
            cache: _,
        } = *self;

        let shape = a.dim();

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
                let (n_range, n_unclipped) = flipped_bounded_range_around(i, dim_i, threshold);
                let (m_range, m_unclipped) = flipped_bounded_range_around(j, dim_j, threshold);
                let fallback = || {
                    let mut sum = 0.;
                    for n in n_range.clone() {
                        for m in m_range.clone() {
                            sum += a[(n as usize, m as usize)] * self.at(i, j, n, m)
                        }
                    }
                    sum
                };
                let sum = if n_unclipped && m_unclipped {
                    match m_range.len() {
                        2 => self.mult_and_sum::<2>(n_range, m_range.start, &a),
                        4 => self.mult_and_sum::<4>(n_range, m_range.start, &a),
                        6 => self.mult_and_sum::<6>(n_range, m_range.start, &a),
                        8 => self.mult_and_sum::<8>(n_range, m_range.start, &a),
                        10 => self.mult_and_sum::<10>(n_range, m_range.start, &a),
                        12 => self.mult_and_sum::<12>(n_range, m_range.start, &a),
                        14 => self.mult_and_sum::<14>(n_range, m_range.start, &a),
                        16 => self.mult_and_sum::<16>(n_range, m_range.start, &a),
                        18 => self.mult_and_sum::<18>(n_range, m_range.start, &a),
                        20 => self.mult_and_sum::<20>(n_range, m_range.start, &a),
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
        unsafe { c.assume_init() }
    }
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

                    // Python logic: if abs(lat[i,j] - lat[n,m]) <= threshold and abs(lon[i,j] - lon[n,m]) <= threshold:
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

// --- PyO3 wrapper functions ---

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
pub fn dispersal_distances_coord_rs<'py>(
    py: Python<'py>,
    length: u32,
    lambda_0: Float,
    // Use PyReadonlyArray to accept a NumPy array from Python safely
    // ArrayView2<Float> means a 2D array of Float elements
    lat: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 2]>>,
    lon: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 2]>>,
    threshold: Float,
) -> PyResult<Bound<'py, PyAny>> {
    // 1. Convert PyReadonlyArray to ArrayView2 for use in the core Rust function
    let lat_view = lat.as_array();
    let lon_view = lon.as_array();

    // 2. Call the core logic
    let result_coo = dispersal_distances_coord(length, lambda_0, lat_view, lon_view, threshold);

    // 3. Convert the Coo sparse structure back to a Python object
    // Assuming `to_python_sparse` converts the Coo struct to a compatible sparse format
    // (e.g., a dictionary of coordinates and values, or a SciPy sparse matrix).
    result_coo.to_python_sparse(py)
}

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

    let (n, o, p) = h.dim();
    if n < n_species {
        panic!("n_species {n_species} is higher than the number of subarrays {n} in h")
    }

    let mut res = Array3::<Float>::uninit((n_species, o, p));
    let ddt = DispersalDistancesThreshold::new(lambda_0_init, threshold);
    res.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, res)| {
            ddt.map(|x| x.powf(1. / lambda_0[i]))
                .apply_to(h.index_axis(Axis(0), i), res);
        });

    PyArray3::from_owned_array(py, unsafe { res.assume_init() })
}

/// Export to Python
#[pymodule]
mod captain2rs {
    #[pymodule_export]
    use super::dispersal_distances_coord_rs;
    #[pymodule_export]
    use super::dispersal_distances_threshold_rs;
    #[pymodule_export]
    use super::dispersal_distances_threshold_test_rs;
    #[pymodule_export]
    use super::num_candidates_rs;
    #[pymodule_export]
    use super::DispersalDistancesThreshold;
}

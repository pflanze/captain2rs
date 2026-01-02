use std::fmt::Debug;
use std::mem::{transmute, MaybeUninit};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayBase, ArrayView2, ArrayViewMut2, Axis, Dim, ViewRepr};
use numpy::{
    pyo3::{prelude::*, pyclass},
    PyArray2, PyArray3, PyReadonlyArray,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::biodivsim::div::{square, FlippedBoundedRangesAround, Float};
use crate::dump::perhaps_dump_iteration_i;

#[macro_export]
macro_rules! _dispersal_dispatch {
    { $threshold:expr, { $($before:tt)* } { $($after:tt)* } } => {
        match $threshold {
            // You can add more specializations if you need a larger
            // `threshold`.
            1 => $($before)* 1 $($after)*,
            2 => $($before)* 2 $($after)*,
            3 => $($before)* 3 $($after)*,
            4 => $($before)* 4 $($after)*,
            5 => $($before)* 5 $($after)*,
            6 => $($before)* 6 $($after)*,
            7 => $($before)* 7 $($after)*,
            8 => $($before)* 8 $($after)*,
            9 => $($before)* 9 $($after)*,
            10 => $($before)* 10 $($after)*,
            11 => $($before)* 11 $($after)*,
            12 => $($before)* 12 $($after)*,
            13 => $($before)* 13 $($after)*,
            14 => $($before)* 14 $($after)*,
            15 => $($before)* 15 $($after)*,
            16 => $($before)* 16 $($after)*,
            _ => panic!(
                "don't have a specialization for threshold {}, please edit the source to add one",
                $threshold
            ),
        }
    }
}

/// The length of the threshold square--symmetric with `threshold`
/// points around the coordinate, i.e. times 2 plus 1.
pub(crate) const fn len_from_threshold(threshold: usize) -> usize {
    threshold * 2 + 1
}

pub trait RowSlice<T> {
    fn row_slice(&self, row_i: usize) -> &[T];
}

impl<'t> RowSlice<Float> for ArrayView2<'t, Float> {
    #[inline]
    fn row_slice(&self, row_i: usize) -> &[Float] {
        let row = self.row(row_i);
        let row = row.as_slice().expect("standard layout");
        // XX ugly, ndarray creates a new struct with a lifetime, but
        // then for .as_slice() it takes &self's lifetime, not the one
        // in the struct. Probably a bug?
        unsafe {
            // Safe because it's just a bug of missing a lifetime in
            // ndarray? XX verify more deeply?
            transmute(row)
        }
    }
}

impl<'t, R: AsRef<[Float]>> RowSlice<Float> for &[R] {
    #[inline]
    fn row_slice(&self, row_i: usize) -> &[Float] {
        self[row_i].as_ref()
    }
}

/// Reimplementation of `dispersal_distances_threshold`, calculating
/// only a single threshold area and re-using it for all points in
/// lookups instead of calculating a sparse array with an area for all
/// input points.
#[pyclass(frozen)]
#[derive(Debug)]
pub struct Dispersal {
    pub threshold: usize,
    pub neg_exp_rate: Float,
    pub cache: Array2<Float>,
}

fn dist_value(neg_exp_rate: Float, dx: i32, dy: i32) -> Float {
    (neg_exp_rate * Float::sqrt((square(dx) + square(dy)) as Float)).exp()
}

#[pymethods]
impl Dispersal {
    #[new]
    pub fn new(lambda_0: Float, threshold: usize) -> Self {
        let neg_exp_rate = -1.0 / lambda_0;
        let length = len_from_threshold(threshold);
        let mut cache = Array2::zeros((length, length));
        for i in 0..length {
            for j in 0..length {
                cache[(i, j)] = dist_value(
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

    fn precise_at(&self, i: usize, j: usize, n: usize, m: usize) -> Float {
        dist_value(self.neg_exp_rate, i as i32 - n as i32, j as i32 - m as i32)
    }

    /// Only valid in range of differences as per `self.threshold`,
    /// panics otherwise
    #[inline]
    fn cached_at(&self, i: usize, j: usize, n: usize, m: usize) -> Float {
        self.cache[(self.threshold + i - n, self.threshold + j - m)]
    }

    /// Run both `cached_at` and `precise_at` and assert equivalence
    /// for testing.
    fn asserting_at(&self, i: usize, j: usize, n: usize, m: usize) -> Float {
        let val1 = self.precise_at(i, j, n, m);
        let val2 = self.cached_at(i, j, n, m);
        assert_eq!(val1, val2);
        val2
    }

    /// Only valid in range of differences as per `self.threshold`,
    /// may panic otherwise
    #[inline]
    fn at(&self, i: usize, j: usize, n: usize, m: usize) -> Float {
        #[cfg(debug_assertions)]
        let w = self.asserting_at(i, j, n, m);
        #[cfg(not(debug_assertions))]
        let w = self.cached_at(i, j, n, m);
        w
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

impl Dispersal {
    // XX careful: once automatic derivation of threshold from lambda0
    // is done, this will not be valid anymore! todo: probably useless
    // and should be removed anyway.
    pub fn map(&self, dot_transform: impl Fn(Float) -> Float) -> Self {
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

    /// Used internally. `a.is_standard_layout()` must be true!
    #[inline]
    fn _dot<const THRESHOLD: usize>(
        &self,
        n_range: Range<usize>,
        m_start: usize,
        a: impl RowSlice<Float>,
    ) -> Float {
        let threshold_len = len_from_threshold(THRESHOLD);
        let mut sum = 0.;
        let n_range_start = n_range.start;
        for n in n_range {
            let a_slice = &a.row_slice(n)[m_start..];
            assert!(a_slice.len() >= threshold_len);
            let cache_row = self.cache.row(n - n_range_start);
            let cache_slice = &cache_row.as_slice().unwrap();
            assert!(cache_slice.len() >= threshold_len);
            for _m in 0..threshold_len {
                sum += a_slice[_m] * cache_slice[_m];
            }
        }
        sum
    }

    pub fn dot<const THRESHOLD: usize>(
        &self,
        area: FlippedBoundedRangesAround,
        a: impl RowSlice<Float> + Debug,
    ) -> Float {
        let FlippedBoundedRangesAround {
            n,
            m,
            n_range,
            n_is_unclipped,
            m_range,
            m_is_unclipped,
        } = area;
        if n_is_unclipped && m_is_unclipped {
            self._dot::<THRESHOLD>(n_range, m_range.start, a)
        } else {
            let mut sum = 0.;
            for n_ in n_range.clone() {
                let row = a.row_slice(n_);
                for m_ in m_range.clone() {
                    sum += row[m_] * self.at(n, m, n_, m_)
                }
            }
            sum
        }
    }

    fn convolve<const THRESHOLD: usize>(
        &self,
        a: ArrayView2<Float>,
        mut c: ArrayViewMut2<MaybeUninit<Float>>,
    ) {
        let Dispersal {
            threshold,
            neg_exp_rate: _,
            cache: _,
        } = *self;

        // Logically, apply the following (where b is the sparse array):

        // for p in 0..shape.0 {
        //     for q in 0..shape.1 {
        //         nm += a[(p, q)] * b.at(p, q);
        //     }
        // }

        assert!(a.is_standard_layout());
        assert!(self.cache.is_standard_layout());

        let (dim_i, dim_j) = a.dim();

        // vertical
        for i in 0..dim_i {
            // horizontal
            for j in 0..dim_j {
                let area = FlippedBoundedRangesAround::new(i, dim_i, j, dim_j, threshold);
                let sum = self.dot::<THRESHOLD>(area, a);
                c[(i, j)].write(sum);
            }
        }
    }

    /// Calculate the equivalent of `einsum("ij,ijnm->nm", a, self)`
    /// and write the result to `c`. Initializes/overwrites all values
    /// in `c`
    pub fn apply_to(&self, a: ArrayView2<Float>, c: ArrayViewMut2<MaybeUninit<Float>>) {
        // Call specializations of the convolve method that carries
        // out the application.
        _dispersal_dispatch!(self.threshold, { self.convolve::< } { >(a, c) })
    }

    pub fn apply(&self, a: ArrayView2<Float>) -> Array2<Float> {
        let mut c = Array2::<Float>::uninit(a.dim());
        self.apply_to(a, c.view_mut());
        unsafe {
            // Safe because `.apply_to` overwrites all values in `c`
            // without reading from it, and Float is Copy
            c.assume_init()
        }
    }
}

static ITERATION: AtomicU64 = AtomicU64::new(0);

/// Equivalent of
///
/// ```python
///     NumCandidates = np.array(
///           [sparse.einsum("ij,ijnm->nm", self._h[i],
///                          self._dumping_dist ** (1 / self._lambda_0[i])
///                    ).todense() for i in range(self._n_species)])
/// ```
///
/// if `self._dumping_dist` is coming from
///
/// ```python
///      dispersal_distances_threshold_rs(
///          self._h[i].shape[0],
///          lambda_0_init,
///          threshold)
/// ```
///
#[pyfunction]
pub fn num_candidates_rs<'py>(
    py: Python<'py>,
    n_species: usize,
    lambda_0_init: Float,
    threshold: usize,
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

    let iteration = ITERATION.fetch_add(1, Ordering::SeqCst);

    let mut result = Array3::<Float>::uninit((n_species, o, p));
    let ddt = Dispersal::new(lambda_0_init, threshold);
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, res)| {
            let local_h: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>> = h.index_axis(Axis(0), i);
            perhaps_dump_iteration_i(iteration, i, local_h, 0. ..25.6);
            ddt.map(|x| x.powf(1. / lambda_0[i])).apply_to(local_h, res);
        });

    // Safe because we iterate over all of axis 0, then overwrite all
    // values in the iterated-over matrices via `.apply_to`, leaving
    // nothing uninitialized.
    PyArray3::from_owned_array(py, unsafe { result.assume_init() })
}

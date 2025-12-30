use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayBase, ArrayView2, ArrayViewMut2, Axis, Dim, ViewRepr};
use numpy::{
    pyo3::{prelude::*, pyclass},
    PyArray2, PyArray3, PyReadonlyArray,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::biodivsim::div::{flipped_bounded_range_around, square, Float};
use crate::dump::perhaps_dump_iteration_i;

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
        let length = threshold * 2;
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

impl Dispersal {
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
    fn dot<const M: usize>(
        &self,
        n_range: Range<usize>,
        m_start: usize,
        a: &ArrayView2<Float>,
    ) -> Float {
        let mut sum = 0.;
        let n_range_start = n_range.start;
        for n in n_range {
            let a_row = a.row(n);
            let a_slice = &a_row.as_slice().unwrap()[m_start..];
            assert!(a_slice.len() >= M);
            let cache_row = self.cache.row(n - n_range_start);
            let cache_slice = &cache_row.as_slice().unwrap();
            assert!(cache_slice.len() >= M);
            for _m in 0..M {
                sum += a_slice[_m] * cache_slice[_m];
            }
        }
        sum
    }

    fn convolve<const M: usize>(
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

        for i in 0..dim_i {
            for j in 0..dim_j {
                let (n_range, n_is_unclipped) = flipped_bounded_range_around(i, dim_i, threshold);
                let (m_range, m_is_unclipped) = flipped_bounded_range_around(j, dim_j, threshold);

                let sum = if n_is_unclipped && m_is_unclipped {
                    self.dot::<M>(n_range, m_range.start, &a)
                } else {
                    let mut sum = 0.;
                    for n in n_range {
                        for m in m_range.clone() {
                            sum += a[(n, m)] * self.at(i, j, n, m)
                        }
                    }
                    sum
                };
                c[(i, j)].write(sum);
            }
        }
    }

    /// Calculate the equivalent of `einsum("ij,ijnm->nm", a, self)`
    /// and write the result to `c`. Initializes/overwrites all values
    /// in `c`
    pub fn apply_to(&self, a: ArrayView2<Float>, c: ArrayViewMut2<MaybeUninit<Float>>) {
        match self.threshold {
            // Specializations of the convolve method that carries out
            // the application. You can add more specializations if
            // you need a larger `threshold`. The `M` parameter on the
            // right is the whole width of a slice, i.e. twice the
            // threshold.
            1 => self.convolve::<2>(a, c),
            2 => self.convolve::<4>(a, c),
            3 => self.convolve::<6>(a, c),
            4 => self.convolve::<8>(a, c),
            5 => self.convolve::<10>(a, c),
            6 => self.convolve::<12>(a, c),
            7 => self.convolve::<14>(a, c),
            8 => self.convolve::<16>(a, c),
            9 => self.convolve::<18>(a, c),
            10 => self.convolve::<20>(a, c),
            11 => self.convolve::<22>(a, c),
            12 => self.convolve::<24>(a, c),
            13 => self.convolve::<26>(a, c),
            14 => self.convolve::<28>(a, c),
            15 => self.convolve::<30>(a, c),
            _ => panic!(
                "don't have a specialization for threshold {}, please edit the source to add one",
                self.threshold
            ),
        }
    }

    pub fn apply(&self, a: ArrayView2<Float>) -> Array2<Float> {
        let mut c = Array2::<Float>::uninit(a.dim());
        self.apply_to(a, c.view_mut());
        // Safe because `.apply_to` overwrites all values in `c`
        unsafe { c.assume_init() }
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

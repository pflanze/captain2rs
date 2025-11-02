use crate::coo::Coo;
use ndarray::{Array2, ArrayView2};
use numpy::{
    pyo3::{self, prelude::*},
    PyArray2, PyReadonlyArray,
};
use pyo3::pyclass;

fn square_i32(x: i32) -> i32 {
    x * x
}

type Float = f32;

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
            let n_min = if i >= threshold { i - threshold } else { 0 };
            let n_max = u32::min(length, i + threshold);

            let m_min = if j >= threshold { j - threshold } else { 0 };
            let m_max = u32::min(length, j + threshold);

            for n in n_min..n_max {
                for m in m_min..m_max {
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
pub struct DispersalDistancesThreshold {
    pub length: u32,
    pub threshold: u32,
    pub neg_exp_rate: Float,
}

#[pymethods]
impl DispersalDistancesThreshold {
    #[new]
    fn new(length: u32, lambda_0: Float, threshold: u32) -> Self {
        Self {
            length,
            threshold,
            neg_exp_rate: -1.0 / lambda_0,
        }
    }

    fn precise_at(&self, i: u32, j: u32, n: u32, m: u32) -> Float {
        let dx = square_i32(i as i32 - n as i32);
        let dy = square_i32(j as i32 - m as i32);
        let dist = Float::sqrt((dx + dy) as Float);
        (self.neg_exp_rate * dist).exp()
    }
}

/// Variant that includes *some* example calculation
fn dispersal_distances_threshold_eval_1(
    // Matrix A
    a: ArrayView2<Float>,
    // Parameters for 4D-tensor B
    // length: u32,
    // lambda_0: Float,
    // threshold: u32,
    // let b = DispersalDistancesThreshold::new(length, lambda_0, threshold);
    b: &DispersalDistancesThreshold,
) -> Array2<Float> {
    let shape = a.dim();
    let mut c = Array2::zeros(shape); // XX use numpy directly?

    // for p in 0..shape.0 {
    //     for q in 0..shape.1 {
    //         nm += a[(p, q)] * b.precise_at(p, q);
    //     }
    // }
    let DispersalDistancesThreshold {
        length,
        threshold,
        neg_exp_rate: _,
    } = *b;
    let dim_i = shape.0 as u32;
    let dim_j = shape.1 as u32;

    (0..dim_i).for_each(|i| {
        // XXX wrong?
        for j in 0..dim_j {
            let n_min = if i >= threshold { i - threshold } else { 0 };
            let n_max = u32::min(length, i + threshold);
            let m_min = if j >= threshold { j - threshold } else { 0 };
            let m_max = u32::min(length, j + threshold);

            for n in n_min..n_max {
                for m in m_min..m_max {
                    c[(n as usize, m as usize)] += a[(i as usize, j as usize)]
                        * b.precise_at(i, j, n, m)
                            // XX hack: user space calculation inlined
                            .powf(1. / 2.3);
                }
            }
        }
    });

    c
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
    println!("calculating distances with coordinate threshold...");

    let mut dumping_dist = Coo::new(0.0);
    let exp_rate = 1.0 / lambda_0;

    // The input arrays must have dimensions (length, length)
    assert_eq!(lat.shape(), &[length as usize, length as usize]);
    assert_eq!(lon.shape(), &[length as usize, length as usize]);

    let len_usize = length as usize;

    for i in 0..len_usize {
        for j in 0..len_usize {
            let lat_ij = *lat.get((i, j)).expect("i, j index out of bounds");
            let lon_ij = *lon.get((i, j)).expect("i, j index out of bounds");

            for n in 0..len_usize {
                for m in 0..len_usize {
                    let lat_nm = *lat.get((n, m)).expect("n, m index out of bounds");
                    let lon_nm = *lon.get((n, m)).expect("n, m index out of bounds");

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
fn dispersal_distances_threshold_eval_1_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray<'py, Float, ndarray::Dim<[usize; 2]>>,
    b: &Bound<'_, DispersalDistancesThreshold>,
) -> Bound<'py, PyArray2<Float>> {
    let a_view = a.as_array();
    let res: Array2<Float> = dispersal_distances_threshold_eval_1(a_view, &b.borrow());
    PyArray2::from_owned_array(py, res)
}

/// Export to Python
#[pymodule]
mod captain2rs {
    #[pymodule_export]
    use super::dispersal_distances_coord_rs;
    #[pymodule_export]
    use super::dispersal_distances_threshold_eval_1_rs;
    #[pymodule_export]
    use super::dispersal_distances_threshold_rs;
    #[pymodule_export]
    use super::dispersal_distances_threshold_test_rs;
    #[pymodule_export]
    use super::DispersalDistancesThreshold;
}

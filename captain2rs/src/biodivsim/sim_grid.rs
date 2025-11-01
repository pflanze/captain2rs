use crate::coo::Coo;
use ndarray::ArrayView2;
use numpy::{pyo3, pyo3::prelude::*, PyReadonlyArray};

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
    lambda_0: f64,
    threshold: u32,
    test_hack: bool,
    default: Option<f64>,
) -> Coo<i64, 4, f64> {
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
                    let dx = (i as f64 - n as f64).powi(2);
                    let dy = (j as f64 - m as f64).powi(2);
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
    lambda_0: f64,
    lat: ArrayView2<f64>,
    lon: ArrayView2<f64>,
    threshold: f64,
) -> Coo<i64, 4, f64> {
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
    lambda_0: f64,
    threshold: u32,
    test_hack: bool,
    default: Option<f64>,
) -> pyo3::PyResult<Bound<'py, PyAny>> {
    dispersal_distances_threshold(length, lambda_0, threshold, test_hack, default)
        .to_python_sparse(py)
}

#[pyfunction]
pub fn dispersal_distances_threshold_rs<'py>(
    py: Python<'py>,
    length: u32,
    lambda_0: f64,
    threshold: u32,
) -> pyo3::PyResult<Bound<'py, PyAny>> {
    dispersal_distances_threshold(length, lambda_0, threshold, false, None).to_python_sparse(py)
}

#[pyfunction]
pub fn dispersal_distances_coord_rs<'py>(
    py: Python<'py>,
    length: u32,
    lambda_0: f64,
    // Use PyReadonlyArray to accept a NumPy array from Python safely
    // ArrayView2<f64> means a 2D array of f64 elements
    lat: PyReadonlyArray<'py, f64, ndarray::Dim<[usize; 2]>>,
    lon: PyReadonlyArray<'py, f64, ndarray::Dim<[usize; 2]>>,
    threshold: f64,
) -> pyo3::PyResult<Bound<'py, PyAny>> {
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

/// Export to Python
#[pymodule(name = "captain2rs")]
fn captain2rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dispersal_distances_threshold_test_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dispersal_distances_threshold_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dispersal_distances_coord_rs, m)?)?;
    Ok(())
}

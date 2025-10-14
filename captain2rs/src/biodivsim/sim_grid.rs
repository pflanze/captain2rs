use ndarray::Array1;
use numpy::{pyo3, pyo3::prelude::*};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::coo::Coo;

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
                        .expect("sorted");
                }
            }
        }
    }

    if test_hack {
        dumping_dist.insert_unordered([5, 9, 7, 1], -7.3456);
    }

    dumping_dist
}

#[pyfunction]
pub fn dispersal_distances_threshold_rs<'py>(
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

#[pymodule(name = "captain2rs")]
fn captain2rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dispersal_distances_threshold_rs, m)?)
}

// def add_random_error(probs, sig=0.1):
//     rates = -np.log(1 - probs)
//     log_rates = np.log(rates)
//     tmp_log_rates = np.random.normal(0, sig * log_rates, probs.shape)
//     rnd_log_rates = log_rates + tmp_log_rates
//     probs = 1 - np.exp(-np.exp(rnd_log_rates))
//     probs = np.maximum(probs, np.zeros(rates.shape) + small_number)
//     probs = np.minimum(probs, np.ones(rates.shape) - small_number)
//     return probs

/// Add random error to probabilities, similar to the Python version.
///
/// # Arguments
/// * `probs` - input probabilities (ndarray of f64)
/// * `sig` - noise scale (default 0.1)
/// * `small_number` - clamp value to avoid exact 0 or 1
pub fn add_random_error(probs: &Array1<f64>, sig: f64, small_number: f64) -> Array1<f64> {
    // rates = -log(1 - probs)
    let rates = probs.mapv(|p| -(1.0 - p).ln());

    // log_rates = log(rates)
    let log_rates = rates.mapv(|r| r.ln());

    // tmp_log_rates = Normal(0, sig * log_rates)
    let mut rng = thread_rng();
    let tmp_log_rates = log_rates.mapv(|lr| {
        let sigma = sig * lr.abs(); // ensure nonnegative stddev
        let normal = Normal::new(0.0, sigma.max(1e-12)).unwrap();
        normal.sample(&mut rng)
    });

    // rnd_log_rates = log_rates + tmp_log_rates
    let rnd_log_rates = &log_rates + &tmp_log_rates;

    // probs = 1 - exp(-exp(rnd_log_rates))
    let mut new_probs = rnd_log_rates.mapv(|rlr| 1.0 - (-rlr.exp()).exp());

    // clamp to [small_number, 1 - small_number]
    new_probs.mapv_inplace(|p| {
        if p < small_number {
            small_number
        } else if p > 1.0 - small_number {
            1.0 - small_number
        } else {
            p
        }
    });

    new_probs
}

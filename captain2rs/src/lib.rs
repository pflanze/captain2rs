pub mod biodivsim;
pub mod coo;
pub mod coo_numpy;
pub mod dump;

use numpy::pyo3::prelude::*;

#[pymodule]
mod captain2rs {
    use numpy::{pyo3::prelude::*, PyReadonlyArray};

    use super::biodivsim::sim_grid::{
        dispersal_distances_coord, dispersal_distances_threshold, Float,
    };

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
    use super::biodivsim::sim_grid::num_candidates_rs;
    #[pymodule_export]
    use super::biodivsim::sim_grid::DispersalDistancesThreshold;
}

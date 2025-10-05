use std::fmt::Debug;

use ndarray::Array2;
use num_traits::PrimInt;
use numpy::{
    pyo3::{
        prelude::*,
        types::{PyAnyMethods, PyDict},
        Bound, IntoPyObject, PyAny, PyResult, Python,
    },
    PyArray1, PyArray2,
};

use crate::coo::Coo;

impl<C: PrimInt + Debug + numpy::Element, const D: usize, V> Coo<C, D, V> {
    /// Convert into a Python `sparse` package object
    pub fn to_python_sparse<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>
    where
        V: Clone + numpy::Element + IntoPyObject<'py>,
        pyo3::PyErr: From<<V as IntoPyObject<'py>>::Error>,
    {
        let (coords, data): ([Vec<C>; D], Vec<V>) = self.to_coords_and_values();
        let nnz = data.len();

        // Build a 2D array of shape (D, nnz)
        let mut coords_arr = Array2::<C>::zeros((D, nnz));
        for (axis, vec) in coords.into_iter().enumerate() {
            coords_arr
                .slice_mut(ndarray::s![axis, ..])
                .assign(&ndarray::Array1::from(vec));
        }

        let coords_py = PyArray2::from_owned_array(py, coords_arr);
        let data_py = PyArray1::from_vec(py, data);

        dbg!(&coords_py);
        dbg!(&data_py);

        // `sparse.COO((coords, data), fill_value = ...)`
        let sparse = py.import("sparse")?;
        let d = PyDict::new(py);
        d.set_item("fill_value", self.fill_value().clone())?;
        sparse.getattr("COO")?.call((coords_py, data_py), Some(&d))
    }
}

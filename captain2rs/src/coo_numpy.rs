use std::fmt::Debug;

use ndarray::Array2;
use num_traits::PrimInt;
use numpy::{
    pyo3::{types::PyAnyMethods, Bound, PyAny, PyResult, Python},
    PyArray1, PyArray2,
};

use crate::coo::Coo;

impl<C: PrimInt + Debug + numpy::Element, const D: usize, V: Copy + numpy::Element> Coo<C, D, V> {
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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

        let sparse = py.import("sparse")?;
        // XXX pass fill_value!
        // assert_eq!(self.fill_value(), &0.);
        sparse.getattr("COO")?.call1((coords_py, data_py))
    }
}

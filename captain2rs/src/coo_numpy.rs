use std::fmt::Debug;

use ndarray::ArrayD;
use num_traits::PrimInt;
use numpy::{
    pyo3::{
        prelude::*,
        types::{PyAnyMethods, PyDict},
        Bound, IntoPyObject, PyAny, PyResult, Python,
    },
    PyArray1, PyArrayDyn,
};

use crate::coo::Coo;

impl<C: PrimInt + Debug + numpy::Element, const D: usize, V> Coo<C, D, V> {
    /// Convert into a Python `sparse` package object
    pub fn to_python_sparse<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>
    where
        V: Clone + numpy::Element + IntoPyObject<'py>,
        pyo3::PyErr: From<<V as IntoPyObject<'py>>::Error>,
    {
        let (coords, data): (Vec<C>, Vec<V>) = self.to_coords_and_values();
        let nnz = data.len();

        let coords_ary = ArrayD::from_shape_vec(vec![D, nnz], coords).expect("XXX");
        let coords_py = PyArrayDyn::from_owned_array(py, coords_ary);
        let data_py = PyArray1::from_vec(py, data);

        // dbg!(&coords_py);
        // dbg!(&data_py);

        // `sparse.COO((coords, data), fill_value = ...)`
        let sparse = py.import("sparse")?;
        let d = PyDict::new(py);
        d.set_item("fill_value", self.fill_value().clone())?;
        sparse.getattr("COO")?.call((coords_py, data_py), Some(&d))
    }
}

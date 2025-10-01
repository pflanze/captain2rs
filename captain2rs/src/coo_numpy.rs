use std::fmt::Debug;

use ndarray::FixedInitializer;
use num_traits::PrimInt;
use numpy::{
    pyo3::{types::PyAnyMethods, Bound, PyAny, PyResult, Python},
    PyArray1, PyArray2,
};

use crate::coo::Coo;

// pub trait CooToNumPy {
//     fn to_numpy(&self) ->
// }

// impl CooToNumPy for Coo {
//     fn to_numpy(&self) -> {

//     }
// }

// XX move to trait to allow as separate crate
impl<C: PrimInt + Debug, const D: usize, V: Copy + numpy::Element> Coo<C, D, V>
where
    [C; D]: FixedInitializer,
    <[C; D] as FixedInitializer>::Elem: numpy::Element,
{
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (coords, data) = self.to_coords_and_values();
        let coords_py = PyArray2::from_owned_array(py, coords.into());
        let data_py = PyArray1::from_vec(py, data);
        dbg!(&coords_py);
        dbg!(&data_py);

        let sparse = py.import("sparse")?;
        sparse.getattr("COO")?.call1((coords_py, data_py))
    }
}

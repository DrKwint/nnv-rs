extern crate ndarray;
extern crate num;
pub mod affine;
pub mod constellation;
pub mod polytope;
pub mod star;
pub mod util;
use affine::Affine;
use constellation::Constellation;
use numpy::PyArray1;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use star::Star;

pub fn build_constellation(py_affines: &PyList) -> Constellation<f64> {
    let ro_affines: Vec<(PyReadonlyArray2<f32>, PyReadonlyArray1<f32>)> =
        py_affines.extract().unwrap();
    let input_dim = ro_affines[0].0.shape()[0];
    let star = Star::default(input_dim);
    let affines: Vec<Affine<f64>> = ro_affines
        .iter()
        .map(|x| {
            Affine::new(
                x.0.as_array().to_owned().mapv(|x| x.into()),
                x.1.as_array().to_owned().mapv(|x| x.into()),
            )
        })
        .collect();
    Constellation::new(star, affines)
}

#[pyfunction]
pub fn sample_constellation(
    py_affines: &PyList,
    loc: PyReadonlyArray1<f64>,
    scale: PyReadonlyArray2<f64>,
    cdf_samples: usize,
    num_samples: usize,
) -> Py<PyList> {
    let mut constellation = build_constellation(py_affines);

    let loc = loc.as_array().to_owned();
    let scale = scale.as_array().to_owned();
    let samples = constellation.sample(loc, scale, cdf_samples, num_samples);
    let gil = Python::acquire_gil();
    let py = gil.python();
    let py_samples: Vec<Py<PyArray1<f64>>> = samples
        .into_iter()
        .map(|x| PyArray1::from_array(py, &x).to_owned())
        .collect();
    let out = PyList::new(py, py_samples);
    out.into_py(py)
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn nnv_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_constellation, m)?)?;
    Ok(())
}

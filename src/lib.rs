extern crate highs;
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
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use star::Star;

pub fn build_constellation(
    py_affines: &PyList,
    input_lower_bounds: Option<PyReadonlyArray1<f64>>,
    input_upper_bounds: Option<PyReadonlyArray1<f64>>,
) -> Constellation<f64> {
    let ro_affines: Vec<(PyReadonlyArray2<f32>, PyReadonlyArray1<f32>)> =
        py_affines.extract().unwrap();
    let input_dim = ro_affines[0].0.shape()[0];
    let mut star = Star::default(input_dim);
    if input_lower_bounds.is_some() {
        star = star.with_input_bounds(
            input_lower_bounds.unwrap().as_array().to_owned(),
            input_upper_bounds.unwrap().as_array().to_owned(),
        );
    }
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
#[text_signature = "(affines, loc, scale, safe_value, cdf_samples, num_samples, input_lower_bounds, input_upper_bounds /)"]
pub fn sample_constellation(
    py_affines: &PyList,
    loc: PyReadonlyArray1<f64>,
    scale: PyReadonlyArray2<f64>,
    safe_value: f64,
    cdf_samples: usize,
    num_samples: usize,
    input_lower_bounds: Option<PyReadonlyArray1<f64>>,
    input_upper_bounds: Option<PyReadonlyArray1<f64>>,
) -> Py<PyArray1<f64>> {
    let mut constellation = build_constellation(py_affines, input_lower_bounds, input_upper_bounds);

    let loc = loc.as_array().to_owned();
    let scale = scale.as_array().to_owned();
    let mut samples = constellation.sample(&loc, &scale, safe_value, cdf_samples, num_samples);
    while samples.is_empty() {
        samples = constellation.sample(&loc, &scale, safe_value, cdf_samples, num_samples);
    }
    let gil = Python::acquire_gil();
    let py = gil.python();
    let py_samples: Vec<Py<PyArray1<f64>>> = samples
        .into_iter()
        .map(|x| PyArray1::from_array(py, &x).to_owned())
        .collect();
    py_samples[0].clone()
}

#[pymodule]
pub fn nnv_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_constellation, m)?)?;
    Ok(())
}

extern crate good_lp;
extern crate highs;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_stats;
extern crate num;
extern crate rand;
extern crate shh;
extern crate truncnorm;

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

#[pyclass]
struct PyConstellation {
    constellation: Constellation<f64>,
}

impl PyConstellation {
    pub fn new(
        py_affines: &PyList,
        input_lower_bounds: Option<PyReadonlyArray1<f64>>,
        input_upper_bounds: Option<PyReadonlyArray1<f64>>,
    ) -> PyConstellation {
        let ro_affines: Vec<(PyReadonlyArray2<f32>, PyReadonlyArray1<f32>)> =
            py_affines.extract().unwrap();
        let input_dim = ro_affines[0].0.shape()[0];
        let mut star = Star::default(input_dim);
        if let Some((lower_bounds, upper_bounds)) = input_lower_bounds.zip(input_upper_bounds) {
            star = star.with_input_bounds(
                lower_bounds.as_array().to_owned(),
                upper_bounds.as_array().to_owned(),
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
        PyConstellation {
            constellation: Constellation::new(star, affines),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_constellation(
        mut self,
        loc: PyReadonlyArray1<f64>,
        scale: PyReadonlyArray2<f64>,
        safe_value: f64,
        cdf_samples: usize,
        num_samples: usize,
        max_iters: usize,
    ) -> (Py<PyArray1<f64>>, f64, f64) {
        let loc = loc.as_array().to_owned();
        let scale = scale.as_array().to_owned();
        let (samples, branch_logp) = self.constellation.sample_multivariate_gaussian(
            &loc,
            &scale,
            safe_value,
            cdf_samples,
            num_samples,
            max_iters,
        );
        let gil = Python::acquire_gil();
        let py = gil.python();
        (
            PyArray1::from_array(py, &samples[0].0).to_owned(),
            samples[0].1,
            branch_logp,
        )
    }
}

pub fn build_constellation(
    py_affines: &PyList,
    input_lower_bounds: Option<PyReadonlyArray1<f64>>,
    input_upper_bounds: Option<PyReadonlyArray1<f64>>,
) -> Constellation<f64> {
    let ro_affines: Vec<(PyReadonlyArray2<f32>, PyReadonlyArray1<f32>)> =
        py_affines.extract().unwrap();
    let input_dim = ro_affines[0].0.shape()[0];
    let mut star = Star::default(input_dim);
    if let Some((lower_bounds, upper_bounds)) = input_lower_bounds.zip(input_upper_bounds) {
        star = star.with_input_bounds(
            lower_bounds.as_array().to_owned(),
            upper_bounds.as_array().to_owned(),
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

#[allow(clippy::too_many_arguments)]
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
    max_iters: usize,
) -> (Py<PyArray1<f64>>, f64, f64) {
    let mut constellation = build_constellation(py_affines, input_lower_bounds, input_upper_bounds);

    let loc = loc.as_array().to_owned();
    let scale = scale.as_array().to_owned();
    let (samples, branch_logp) = constellation.sample_multivariate_gaussian(
        &loc,
        &scale,
        safe_value,
        cdf_samples,
        num_samples,
        max_iters,
    );
    let gil = Python::acquire_gil();
    let py = gil.python();
    //(PyArray1::from_array(py, &samples[0]).to_owned(), logp[0])
    (
        PyArray1::from_array(py, &samples[0].0).to_owned(),
        samples[0].1,
        branch_logp,
    )
}

#[pymodule]
pub fn nnv_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyConstellation>()?;
    m.add_function(wrap_pyfunction!(sample_constellation, m)?)?;
    Ok(())
}

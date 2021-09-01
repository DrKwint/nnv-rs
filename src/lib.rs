#![allow(clippy::must_use_candidate)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
extern crate env_logger;
extern crate good_lp;
extern crate highs;
extern crate itertools;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_stats;
extern crate num;
#[cfg(test)]
extern crate proptest;
extern crate rand;
extern crate shh;
extern crate truncnorm;

pub mod affine;
mod bounds;
pub mod constellation;
mod deeppoly;
mod dnn;
mod inequality;
pub mod polytope;
pub mod star;
mod star_node;
mod tensorshape;
#[cfg(test)]
mod test_util;
pub mod util;

use crate::affine::Affine2;
use crate::affine::Affine4;
use crate::bounds::Bounds;
use crate::bounds::Bounds1;
use crate::dnn::DNNIndex;
use crate::dnn::DNNIterator;
use crate::dnn::Layer;
use crate::dnn::DNN;
use affine::Affine;
use constellation::Constellation;
use ndarray::Ix2;
use numpy::PyArray1;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::PyObjectProtocol;
use rand::thread_rng;
use star::Star2;

#[pyclass]
#[derive(Clone, Debug)]
struct PyDNN {
    dnn: DNN<f64>,
}

#[pymethods]
impl PyDNN {
    #[new]
    fn new() -> Self {
        Self {
            dnn: DNN::default(),
        }
    }

    pub fn input_shape(&self) -> Vec<Option<usize>> {
        self.dnn.input_shape().into()
    }

    fn add_dense(&mut self, filters: PyReadonlyArray2<f32>, bias: PyReadonlyArray1<f32>) {
        self.dnn.add_layer(Layer::new_dense(Affine2::new(
            filters.as_array().to_owned().mapv(f64::from),
            bias.as_array().to_owned().mapv(f64::from),
        )))
    }

    fn add_conv(&mut self, filters: PyReadonlyArray4<f32>, bias: PyReadonlyArray1<f32>) {
        self.dnn.add_layer(Layer::new_conv(Affine4::new(
            filters.as_array().to_owned().mapv(f64::from),
            bias.as_array().to_owned().mapv(f64::from),
        )))
    }

    fn add_maxpool(&mut self, pool_size: usize) {
        self.dnn.add_layer(Layer::new_maxpool(pool_size))
    }

    fn add_flatten(&mut self) {
        self.dnn.add_layer(Layer::Flatten)
    }

    fn add_relu(&mut self, ndim: usize) {
        self.dnn.add_layer(Layer::new_relu(ndim))
    }

    fn deeppoly_output_bounds(
        &self,
        lower_input_bounds: PyReadonlyArray1<f64>,
        upper_input_bounds: PyReadonlyArray1<f64>,
    ) -> Py<PyTuple> {
        let input_bounds = Bounds1::new(
            lower_input_bounds.as_array().to_owned(),
            upper_input_bounds.as_array().to_owned(),
        );
        let output_bounds = deeppoly::deep_poly(
            input_bounds,
            DNNIterator::new(&self.dnn, DNNIndex::default()),
        );
        let gil = Python::acquire_gil();
        let py = gil.python();
        let out_lbs = PyArray1::from_array(py, &output_bounds.lower());
        let out_ubs = PyArray1::from_array(py, &output_bounds.upper());
        PyTuple::new(py, &[out_lbs, out_ubs]).into()
    }
}

#[pyproto]
impl PyObjectProtocol for PyDNN {
    fn __str__(&self) -> String {
        format!("DNN: {}", self.dnn)
    }
}

#[pyclass]
struct PyConstellation {
    constellation: Constellation<f64, Ix2>,
}

#[pymethods]
impl PyConstellation {
    #[new]
    pub fn new(
        py_dnn: PyDNN,
        input_bounds: Option<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)>,
    ) -> Self {
        let dnn = py_dnn.dnn;
        let input_shape = dnn.input_shape();

        let bounds = input_bounds
            .map(|(lbs, ubs)| Bounds::new(lbs.as_array().to_owned(), ubs.as_array().to_owned()));

        let star = match input_shape.rank() {
            1 => {
                let mut star = Star2::default(&input_shape);
                if let Some(ref b) = bounds {
                    star = star.with_input_bounds((*b).clone());
                }
                star
            }
            _ => {
                panic!()
            }
        };
        Self {
            constellation: Constellation::new(star, dnn, bounds),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bounded_sample_multivariate_gaussian(
        &mut self,
        loc: PyReadonlyArray1<f64>,
        scale: PyReadonlyArray2<f64>,
        safe_value: f64,
        cdf_samples: usize,
        num_samples: usize,
        max_iters: usize,
    ) -> (Py<PyArray1<f64>>, f64, f64) {
        let mut rng = thread_rng();
        let loc = loc.as_array().to_owned();
        let scale = scale.as_array().to_owned();
        let (samples, branch_logp) = self.constellation.bounded_sample_multivariate_gaussian(
            &mut rng,
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

/// # Errors
#[pymodule]
pub fn nnv_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();
    #[cfg(debug_assertions)]
    m.add_class::<PyConstellation>()?;
    m.add_class::<PyDNN>()?;
    Ok(())
}

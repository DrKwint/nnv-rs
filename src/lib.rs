#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::must_use_candidate)]
#![feature(fn_traits)]
#![feature(destructuring_assignment)]
#![feature(unboxed_closures)]
#![feature(trait_alias)]
#![feature(convert_float_to_int)]
extern crate approx;
#[cfg(feature = "openblas-system")]
extern crate blas_src;
extern crate env_logger;
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
pub mod asterism;
pub mod belt;
pub mod bounds;
pub mod constellation;
pub mod deeppoly;
pub mod dnn;
pub mod gaussian;
pub mod inequality;
pub mod lp;
pub mod polytope;
pub mod star;
pub mod star_node;
pub mod tensorshape;
pub mod test_util;
pub mod util;

cfg_if::cfg_if! {
    if #[cfg(feature = "blas_intel-mkl")] {
    } else if #[cfg(feature = "blas_openblas-system")] {
    } else {
        compile_error!("Must enable one of \"blas_{{intel_mkl,optblas-system}}\"");
    }
}

pub trait NNVFloat = 'static
    + num::Float
    + std::convert::From<f64>
    + std::convert::Into<f64>
    + ndarray::ScalarOperand
    + std::fmt::Display
    + std::fmt::Debug
    + std::ops::MulAssign
    + std::ops::AddAssign
    + std::default::Default
    + std::iter::Sum
    + approx::AbsDiffEq
    + rand::distributions::uniform::SampleUniform;

pub mod trunks {
    use ndarray::{Array1, Array2, Axis};
    use crate::polytope::Polytope;
    use crate::bounds::Bounds1;

    pub fn halfspace_gaussian_cdf(coeffs: Array1<f64>, rhs: f64, mu: Array1<f64>, sigma: Array1<f64>) -> f64 {
        let mut rng = rand::thread_rng();
        let bounds = Bounds1::trivial(coeffs.len());
        let polytope = Polytope::new(coeffs.insert_axis(Axis(0)), Array1::from_vec(vec![rhs]), bounds);
        let mut truncnorm = polytope.get_truncnorm_distribution(
            &mu,
            &Array2::from_diag(&sigma),
            3,
            1e-10,
        );
        truncnorm.cdf(1000, &mut rng)
    }
}
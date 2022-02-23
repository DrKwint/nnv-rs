#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::must_use_candidate)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(associated_type_bounds)]
#![feature(generic_associated_types)]

extern crate approx;
#[cfg(feature = "openblas-system")]
extern crate blas_src;
extern crate itertools;
#[macro_use]
extern crate more_asserts;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_stats;
extern crate num;
#[cfg(test)]
extern crate proptest;
extern crate rand;
extern crate serde;
extern crate shh;
extern crate truncnorm;

pub mod affine;
pub mod bounds;
pub mod deeppoly;
pub mod dnn;
pub mod gaussian;
pub mod lp;
pub mod polytope;
pub mod star;
pub mod star_node;
pub mod starsets;
pub mod tensorshape;
pub mod test_util;
pub mod util;

cfg_if::cfg_if! {
    if #[cfg(feature = "blas_intel-mkl")] {
    } else if #[cfg(feature = "blas_openblas-system")] {
    } else {
        compile_error!("Must enable one of \"blas_{{intel_mkl,openblas-system}}\"");
    }
}

pub type NNVFloat = f64;

pub mod trunks {
    use crate::polytope::Polytope;
    use ndarray::{Array1, Array2, Axis};

    pub fn halfspace_gaussian_cdf(
        coeffs: Array1<f64>,
        rhs: f64,
        mu: &Array1<f64>,
        sigma: &Array1<f64>,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let polytope = Polytope::new(coeffs.insert_axis(Axis(0)), Array1::from_vec(vec![rhs]));
        let mut truncnorm = polytope.get_truncnorm_distribution(
            mu.view(),
            Array2::from_diag(sigma).view(),
            3,
            1e-10,
        );
        truncnorm.cdf(1000, &mut rng)
    }
}

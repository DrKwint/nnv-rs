//#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::must_use_candidate)]
#![feature(fn_traits)]
#![feature(destructuring_assignment)]
#![feature(unboxed_closures)]
#![feature(trait_alias)]
#![feature(convert_float_to_int)]
extern crate approx;
extern crate env_logger;
extern crate good_lp;
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
pub mod inequality;
pub mod polytope;
pub mod star;
pub mod star_node;
pub mod tensorshape;
pub mod test_util;
pub mod util;

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

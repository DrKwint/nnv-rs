use crate::affine::Affine2;
use log::debug;
use ndarray::concatenate;
use ndarray::Axis;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};
use num::Float;
use std::ops::Mul;
use std::ops::MulAssign;

#[derive(Clone, Debug)]
pub struct Inequality<T: Float> {
    coeffs: Array2<T>, // Assume rows correspond to equations and cols are vars, i.e. Ax < b
    rhs: Array1<T>,
}

impl<T: 'static + Float> Inequality<T> {
    pub fn new(coeffs: Array2<T>, rhs: Array1<T>) -> Self {
        Self { coeffs, rhs }
    }

    pub fn coeffs(&self) -> ArrayView2<T> {
        self.coeffs.view()
    }

    pub fn rhs(&self) -> ArrayView1<T> {
        self.rhs.view()
    }

    pub fn rhs_mut(&mut self) -> ArrayViewMut1<T> {
        self.rhs.view_mut()
    }

    pub fn num_constraints(&self) -> usize {
        self.rhs.len()
    }

    pub fn add_eqns(&mut self, eqns: &Self) {
        self.coeffs.append(Axis(0), eqns.coeffs.view()).unwrap();
        self.rhs.append(Axis(0), eqns.rhs.view()).unwrap();
    }

    pub fn any_nan(&self) -> bool {
        self.coeffs().iter().any(|x| x.is_nan()) || self.rhs.iter().any(|x| x.is_nan())
    }

    pub fn filter_trivial(&mut self) {
        let (coeffs, rhs): (Vec<ArrayView1<T>>, Vec<_>) = self
            .coeffs
            .rows()
            .into_iter()
            .zip(self.rhs().iter())
            .filter(|(coeffs, _rhs)| !coeffs.iter().all(|x| *x == T::zero()))
            .unzip();
        self.coeffs = ndarray::stack(Axis(0), &coeffs).unwrap();
        self.rhs = Array1::from_vec(rhs);
    }

    /// Assumes that the zero valued
    ///
    /// # Panics
    pub fn reduce_with_values(&self, x: ArrayView1<T>, idxs: ArrayView1<bool>) -> Self {
        let rhs_reduction: Array1<T> = self.coeffs.dot(&x);
        let new_rhs = &self.rhs - rhs_reduction;

        let vars = self.coeffs.columns();
        let new_coeffs: Vec<ArrayView2<T>> = vars
            .into_iter()
            .zip(&idxs)
            .filter(|x| *x.1)
            .map(|x| x.0.insert_axis(Axis(1)))
            .collect();
        let new_eqns = concatenate(Axis(1), new_coeffs.as_slice()).unwrap();
        Self::new(new_eqns, new_rhs)
    }
}

/// Scale by scalar
impl<T: Float + ndarray::ScalarOperand + std::ops::Mul> Mul<T> for Inequality<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            coeffs: &self.coeffs * rhs,
            rhs: &self.rhs * rhs,
        }
    }
}

/// Scale by scalar
impl<T: Float + ndarray::ScalarOperand + std::ops::MulAssign> MulAssign<T> for Inequality<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.coeffs *= rhs;
        self.rhs *= rhs;
    }
}

impl<T: 'static + Float> From<Affine2<T>> for Inequality<T> {
    fn from(aff: Affine2<T>) -> Self {
        Self {
            coeffs: aff.basis().to_owned(),
            rhs: aff.shift().to_owned(),
        }
    }
}
/*
#tests

test_add_eqns

test_reduce_with_values
*/

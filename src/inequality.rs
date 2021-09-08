use crate::affine::Affine2;
use ndarray::concatenate;
use ndarray::Axis;
use ndarray::Slice;
use ndarray::Zip;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2};
use num::Float;
use std::convert::TryFrom;
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

    pub fn num_dims(&self) -> usize {
        self.coeffs.ncols()
    }

    pub fn num_constraints(&self) -> usize {
        self.rhs.len()
    }

    pub fn add_eqns(&mut self, eqns: &Self) {
        self.coeffs.append(Axis(0), eqns.coeffs.view()).unwrap();
        self.rhs.append(Axis(0), eqns.rhs.view()).unwrap();
    }

    pub fn get_eqn(&self, idx: usize) -> Self {
        let i_idx: isize = isize::try_from(idx).unwrap();
        Self {
            coeffs: self
                .coeffs
                .slice_axis(Axis(0), Slice::new(i_idx, Some(i_idx + 1), 1))
                .to_owned(),
            rhs: self
                .rhs
                .slice_axis(Axis(0), Slice::new(i_idx, Some(i_idx + 1), 1))
                .to_owned(),
        }
    }

    pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
        let vals = self.coeffs.dot(point);
        Zip::from(&self.rhs)
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
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

#[cfg(test)]
mod tests {
    use crate::test_util::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_inequality_get_eqn(ineq in inequality(3, 4)) {
            let eqn = ineq.get_eqn(2);
            let coeffs = eqn.coeffs();
            prop_assert_eq!(&coeffs.shape(), &vec![1, 3])
        }
    }
}

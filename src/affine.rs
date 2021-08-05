#![allow(non_snake_case, clippy::module_name_repetitions)]
//! Representation of affine transformations
use crate::tensorshape::TensorShape;
use ndarray::concatenate;
use ndarray::ArrayViewMut1;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::ScalarOperand;
use ndarray::ShapeError;
use ndarray::Zip;
use ndarray::{Array, Array1, Array2, Array4};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{Ix2, Ix4};
use num::Float;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign};

pub type Affine2<A> = Affine<A, Ix2>;
pub type Affine4<A> = Affine<A, Ix4>;

/// Affine map data structure
#[derive(Clone, Debug)]
pub struct Affine<T: Float, D: Dimension> {
	basis: Array<T, D>,
	shift: Array1<T>,
}

impl<T: Float, D: Dimension> Display for Affine<T, D> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
		write!(
			f,
			"Basis {:?} Shift {:?}",
			self.basis.shape(),
			self.shift.shape()
		)
	}
}

impl<T: Float, D: Dimension> Affine<T, D> {
	pub fn ndim(&self) -> usize {
		self.basis.ndim()
	}

	pub fn shift(&self) -> ArrayView1<T> {
		self.shift.view()
	}

	pub fn shift_mut(&mut self) -> ArrayViewMut1<T> {
		self.shift.view_mut()
	}

	pub fn into_dyn(self) -> Affine<T, IxDyn> {
		Affine {
			basis: self.basis.into_dyn(),
			shift: self.shift,
		}
	}
}

impl<T: Float> Affine<T, IxDyn> {
	/// # Errors
	pub fn into_dimensionality<D: Dimension>(self) -> Result<Affine<T, D>, ShapeError> {
		let shift = self.shift;
		self.basis
			.into_dimensionality::<D>()
			.map(|basis| Affine { basis, shift })
	}
}

/// Assumes that the affine is f(x) = Ax + b
impl<T: 'static + Float> Affine2<T> {
	/// # Panics
	/// If improper shapes are passed in
	pub fn new(basis: Array2<T>, shift: Array1<T>) -> Self {
		assert_eq!(basis.shape()[0], shift.len());
		Self { basis, shift }
	}

	pub fn identity(ndim: usize) -> Self {
		Self {
			basis: Array2::eye(ndim),
			shift: Array1::zeros(ndim),
		}
	}

	pub fn basis(&self) -> ArrayView2<T> {
		self.basis.view()
	}

	pub fn input_dim(&self) -> usize {
		self.basis.shape()[1]
	}

	pub fn output_dim(&self) -> usize {
		self.shift.len()
	}

	pub fn shape(&self) -> &[usize] {
		self.basis.shape()
	}

	pub fn zero_eqn(&mut self, idx: usize) {
		self.basis.index_axis_mut(Axis(0), idx).fill(num::zero());
	}

	pub fn get_raw_augmented(&self) -> Array2<T> {
		concatenate![Axis(1), self.basis, self.shift.clone().insert_axis(Axis(0))]
	}

	/// Get a single equation (i.e., a set of coefficients and a shift/RHS)
	pub fn get_eqn(&self, index: usize) -> Self {
		let basis = self
			.basis
			.index_axis(Axis(0), index)
			.to_owned()
			.insert_axis(Axis(0));
		let shift = self
			.shift
			.index_axis(Axis(0), index)
			.to_owned()
			.insert_axis(Axis(0));
		Self { basis, shift }
	}

	pub fn apply(&self, x: &ArrayView1<T>) -> Array1<T> {
		self.basis.dot(x) + &self.shift
	}
}

impl<T: 'static + Float + Mul + ScalarOperand> Affine2<T> {
	pub fn scale_eqns(&mut self, x: ArrayView1<T>) {
		assert_eq!(self.basis.nrows(), x.len());
		Zip::from(self.basis.rows_mut())
			.and(x)
			.for_each(|mut row, &x| row.assign(&(&row * x)))
	}
}

/// Add scalar
impl<T: Float + ScalarOperand + Add, D: Dimension> Add<T> for Affine<T, D> {
	type Output = Self;

	fn add(self, rhs: T) -> Self {
		Self {
			basis: self.basis,
			shift: &self.shift + rhs,
		}
	}
}

/// Add vec
impl<T: Float + ScalarOperand + Add, D: Dimension> Add<Array1<T>> for Affine<T, D> {
	type Output = Self;

	fn add(self, rhs: Array1<T>) -> Self {
		Self {
			basis: self.basis,
			shift: &self.shift + rhs,
		}
	}
}

impl<T: Float + ScalarOperand + AddAssign, D: Dimension> AddAssign<T> for Affine<T, D> {
	fn add_assign(&mut self, rhs: T) {
		self.shift += rhs;
	}
}

/// Scale Affine by scalar
impl<T: Float + ScalarOperand + Mul, D: Dimension> Mul<T> for Affine<T, D> {
	type Output = Self;

	fn mul(self, rhs: T) -> Self {
		Self {
			basis: &self.basis * rhs,
			shift: &self.shift * rhs,
		}
	}
}

/// Scale Affine by vector
impl<T: Float + ScalarOperand + Mul> Mul<Array1<T>> for Affine2<T> {
	type Output = Self;

	fn mul(self, rhs: Array1<T>) -> Self {
		Self {
			basis: &self.basis * &rhs,
			shift: &self.shift * rhs,
		}
	}
}

/// Scale Affine by vector
impl<T: Float + ScalarOperand + MulAssign> MulAssign<Array1<T>> for Affine2<T> {
	fn mul_assign(&mut self, rhs: Array1<T>) {
		self.basis *= &rhs;
		self.shift *= &rhs;
	}
}

/// Scale Affine by scalar
impl<T: Float + ScalarOperand + MulAssign, D: Dimension> MulAssign<T> for Affine<T, D> {
	fn mul_assign(&mut self, scalar: T) {
		self.basis *= scalar;
		self.shift *= scalar;
	}
}

impl<'a, 'b, T: 'static + Float> Mul<&'b Affine2<T>> for &'a Affine2<T> {
	type Output = Affine2<T>;

	#[allow(clippy::suspicious_arithmetic_impl)]
	fn mul(self, rhs: &'b Affine2<T>) -> Affine2<T> {
		let basis = self.basis.dot(&rhs.basis);
		let shift = self.basis.dot(&rhs.shift) + self.shift.clone();
		Affine { basis, shift }
	}
}

/// Apply Affine to Affine
impl<T: Float + ndarray::ScalarOperand + std::ops::Mul> Mul<&Self> for Affine2<T> {
	type Output = Self;

	#[allow(clippy::suspicious_arithmetic_impl)]
	fn mul(self, rhs: &Self) -> Self {
		let basis = self.basis.dot(&rhs.basis);
		let shift = self.basis.dot(&rhs.shift) + self.shift;
		Self { basis, shift }
	}
}

impl<T: 'static + Float> Affine<T, Ix4> {
	pub fn new(basis: Array4<T>, shift: Array1<T>) -> Self {
		Self { basis, shift }
	}

	pub fn output_channels(&self) -> usize {
		self.shift.len()
	}

	pub fn input_shape(&self) -> TensorShape {
		TensorShape::new(vec![None, None, Some(self.basis.shape()[2])])
	}
}

#[cfg(test)]
mod tests {
    use crate::test_util::affine2;
    use crate::test_util::array1;
    use proptest::prelude::*;

    proptest! {

        #[test]
        fn test_affine_composability(start in array1(16), aff_1 in affine2(16, 16), aff_2 in affine2(16, 16)) {
            let result_1 = (&aff_1 * &aff_2).apply(&start.view());
            let result_2 = aff_1.apply(&aff_2.apply(&start.view()).view());
            prop_assert_eq!(result_1, result_2);
        }
    }

    /*
    #[test]
    fn get_eqn_works() {}

    #[test]
    fn apply_works() {}
     */
}

#![allow(non_snake_case, clippy::module_name_repetitions)]
//! Representation of affine transformations
use crate::bounds::Bounds1;
use crate::tensorshape::TensorShape;
use ndarray::concatenate;
use ndarray::iter::Lanes;
use ndarray::ScalarOperand;
use ndarray::ShapeError;
use ndarray::Zip;
use ndarray::{Array, Array1, Array2, Array4};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{ArrayViewMut0, ArrayViewMut1};
use ndarray::{Axis, Dimension};
use ndarray::{Ix1, Ix2, Ix4, IxDyn};
use num::Float;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign};

pub type Affine2<A> = Affine<A, Ix2>;
pub type Affine4<A> = Affine<A, Ix4>;

/// Affine map data structure
#[derive(Clone, Debug, PartialEq)]
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

    pub fn get_eqn_mut(&mut self, index: usize) -> (ArrayViewMut1<T>, ArrayViewMut0<T>) {
        (
            self.basis.index_axis_mut(Axis(0), index),
            self.shift.index_axis_mut(Axis(0), index),
        )
    }

    pub fn vars(&self) -> Lanes<T, Ix1> {
        self.basis.columns()
    }

    pub fn apply(&self, x: &ArrayView1<T>) -> Array1<T> {
        self.basis.dot(x) + &self.shift
    }

    pub fn split_at(&self, index: usize) -> (Self, Self) {
        let (basis_head, basis_tail) = self.basis.view().split_at(Axis(1), index);
        (
            Self {
                basis: basis_head.to_owned(),
                shift: self.shift.clone(),
            },
            Self {
                basis: basis_tail.to_owned(),
                shift: self.shift.clone(),
            },
        )
    }

    pub fn append(mut self, other: Self) -> Self {
        self.basis.append(Axis(1), other.basis.view()).unwrap();
        self
    }
}

impl<T: 'static + Float + Sum + Debug> Affine2<T> {
    pub fn signed_apply(&self, bounds: &Bounds1<T>) -> Bounds1<T> {
        let lower = crate::util::signed_dot(
            &self.basis.view(),
            &bounds.lower().view(),
            &bounds.upper().view(),
        ) + &self.shift;
        let upper = crate::util::signed_dot(
            &self.basis.view(),
            &bounds.upper().view(),
            &bounds.lower().view(),
        ) + &self.shift;
        Bounds1::new(lower, upper)
    }

    pub fn signed_compose(&self, pos_rhs: &Self, neg_rhs: &Self) -> Self {
        assert_eq!(
            self.input_dim(),
            pos_rhs.output_dim(),
            "self input dim: {}, pos_rhs output dim: {}",
            self.input_dim(),
            pos_rhs.output_dim()
        );
        assert_eq!(self.input_dim(), neg_rhs.output_dim());
        Affine2 {
            basis: crate::util::signed_matmul(
                &self.basis.view(),
                &pos_rhs.basis.view(),
                &neg_rhs.basis.view(),
            ),
            shift: &crate::util::signed_dot(
                &self.basis.view(),
                &pos_rhs.shift.view(),
                &neg_rhs.shift.view(),
            ) + &self.shift,
        }
    }
}

impl<T: 'static + Float + Mul + ScalarOperand + std::ops::MulAssign> Affine2<T> {
    pub fn scale_eqns(&mut self, x: ArrayView1<T>) {
        assert_eq!(self.basis.nrows(), x.len());
        Zip::from(self.basis.rows_mut())
            .and(self.shift.view_mut())
            .and(x)
            .for_each(|mut row, shift, &x| {
                row.assign(&(&row * x));
                *shift *= x;
            })
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
    use crate::affine::Affine2;
    use crate::test_util::*;
    use ndarray::Array;
    use ndarray::Array2;
    use ndarray::ArrayView;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_affine_composability(start in array1(4), aff_1 in affine2(2, 3), aff_2 in affine2(4, 2)) {
            let result_1 = (&aff_1 * &aff_2).apply(&start.view());
            let result_2 = aff_1.apply(&aff_2.apply(&start.view()).view());
            prop_assert!(result_1.to_owned().abs_diff_eq(&result_2.to_owned(), 1e-8));
        }

        #[test]
        fn get_eqn_works(aff in affine2(2,3)) {
            for i in 0..3 {
                let eqn = aff.get_eqn(i);
                prop_assert_eq!(aff.basis.row(i), eqn.basis.row(0));
            }
        }

        #[test]
        fn test_signed_apply(mut bounds in bounds1(2), aff in affine2(2, 2)) {
            prop_assert!(bounds.lower().iter().zip(bounds.upper().iter())
                         .all(|(a, b)| a <= b));
            bounds = aff.signed_apply(&bounds);
            prop_assert!(bounds.lower().iter().zip(bounds.upper().iter())
                         .all(|(a, b)| a <= b));
        }

        #[test]
        fn test_signed_compose(bounds in bounds1(2), aff in affine2(2, 2), aff2 in affine2(2, 2)) {
            let lower_aff = Affine2::identity(2);
            let upper_aff = Affine2::identity(2);

            let lower_pre = lower_aff.apply(&bounds.lower());
            let upper_pre = upper_aff.apply(&bounds.upper());

            prop_assert!(lower_pre.iter().zip(upper_pre.iter()).all(|(a, b)| a <= b));

            let lower_aff_2 = aff.signed_compose(&lower_aff, &upper_aff);
            let upper_aff_2 = aff.signed_compose(&upper_aff, &lower_aff);
            let mut lower = lower_aff_2.signed_apply(&bounds);
            let mut upper = upper_aff_2.signed_apply(&bounds);

            prop_assert!(lower.lower().iter().zip(upper.lower().iter()).all(|(a, b)| a <= b));
            prop_assert!(lower.upper().iter().zip(upper.upper().iter()).all(|(a, b)| a <= b));
            prop_assert!(lower.lower().iter().zip(upper.upper().iter()).all(|(a, b)| a <= b));

            let lower_aff_3 = aff2.signed_compose(&lower_aff_2, &upper_aff_2);
            let upper_aff_3 = aff2.signed_compose(&upper_aff_2, &lower_aff_2);
            lower = lower_aff_3.signed_apply(&bounds);
            upper = upper_aff_3.signed_apply(&bounds);

            prop_assert!(lower.lower().iter().zip(upper.lower().iter()).all(|(a, b)| a <= b));
            prop_assert!(lower.upper().iter().zip(upper.upper().iter()).all(|(a, b)| a <= b));
            prop_assert!(lower.lower().iter().zip(upper.upper().iter()).all(|(a, b)| a <= b));
        }
    }
}

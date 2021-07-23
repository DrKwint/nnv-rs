//! Representation of affine transformations
#![allow(non_snake_case)]
use crate::tensorshape::TensorShape;
use ndarray::concatenate;
use ndarray::Array;
use ndarray::Array4;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::ShapeError;
use ndarray::{s, Axis, Slice};
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{ArrayViewMut1, ArrayViewMut2};
use ndarray::{Ix2, Ix4};
use num::Float;
use std::ops::{Mul, MulAssign};

pub type Affine2<A> = Affine<A, Ix2>;
pub type Affine4<A> = Affine<A, Ix4>;

/// Affine map data structure
#[derive(Clone, Debug)]
pub struct Affine<T: Float, D: Dimension> {
    basis: Array<T, D>,
    shift: Array1<T>,
}

impl<T: Float, D: Dimension> Affine<T, D> {
    pub fn ndim(&self) -> usize {
        self.basis.ndim()
    }

    pub fn into_dyn(self) -> Affine<T, IxDyn> {
        Affine {
            basis: self.basis.into_dyn(),
            shift: self.shift,
        }
    }
}

impl<T: Float> Affine<T, IxDyn> {
    pub fn into_dimensionality<D: Dimension>(self) -> Result<Affine<T, D>, ShapeError> {
        let shift = self.shift;
        self.basis
            .into_dimensionality::<D>()
            .map(|basis| Affine { basis, shift })
    }
}

/// Assumes that the affine is f(x) = Ax + b
impl<T: 'static + Float> Affine2<T> {
    pub fn new(basis: Array2<T>, shift: Array1<T>) -> Self {
        assert_eq!(basis.shape()[1], shift.len());
        Self { basis, shift }
    }

    pub fn basis(&self) -> ArrayView2<T> {
        self.basis.view()
    }

    pub fn shift(&self) -> ArrayView1<T> {
        self.shift.view()
    }

    pub fn shift_mut(&mut self) -> ArrayViewMut1<T> {
        self.shift.view_mut()
    }

    pub fn input_dim(&self) -> usize {
        self.basis.shape()[1]
    }

    pub fn output_dim(&self) -> usize {
        self.shift.len()
    }

    pub fn zero_eqn(&mut self, idx: usize) {
        self.basis.index_axis_mut(Axis(0), idx).fill(num::zero());
    }

    pub fn apply(coordinates: Array2<T>) -> Array2<T> {
        todo!()
    }

    /// Get a single equation (i.e., a set of coefficients and a shift/RHS)
    pub fn get_eqn_affine(&self, index: usize) -> Affine2<T> {
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
}

/// Scale Affine by scalar
impl<T: Float + ndarray::ScalarOperand + std::ops::Mul> Mul<T> for Affine2<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            basis: &self.basis * rhs,
            shift: &self.shift * rhs,
        }
    }
}

impl<'a, 'b, T: 'static + Float> Mul<&'b Affine2<T>> for &'a Affine2<T> {
    type Output = Affine2<T>;

    fn mul(self, rhs: &'b Affine2<T>) -> Affine2<T> {
        let basis = self.basis.dot(&rhs.basis);
        let shift = self.basis.dot(&rhs.shift) + self.shift.clone();
        Affine { basis, shift }
    }
}

/// Apply Affine to Affine
impl<T: Float + ndarray::ScalarOperand + std::ops::Mul> Mul<&Affine2<T>> for Affine2<T> {
    type Output = Self;

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

/*
impl<T: 'static + Float> Affine<T, Ix2> {
    /// Instantiate Affine with given matrix and vector
    pub fn new(mul: Array2<T>, shift: Array1<T>) -> Self {
        if mul.ncols() == shift.len() {
            Self {
                matrix: concatenate!(Axis(0), mul, shift.insert_axis(Axis(0))),
                is_lhs: false,
            }
        } else {
            Self {
                matrix: concatenate!(Axis(1), mul, shift.insert_axis(Axis(1))),
                is_lhs: true,
            }
        }
    }

    /// Instantiate Affine with augmented matrix
    ///
    /// `is_lhs` indicates whether affine is $f(x) = Ax+b$ or $f(x) = xA+b$
    pub fn from_raw(raw: Array2<T>, is_lhs: bool) -> Self {
        Self {
            matrix: raw,
            is_lhs,
        }
    }

    /// Get the augmented matrix which backs the data structure
    pub fn get_augmented(&self) -> ArrayView2<T> {
        self.matrix.view()
    }

    /// Dimension of input vector $x$ to the affine $f(x)$
    pub fn input_dim(&self) -> usize {
        if self.is_lhs {
            self.matrix.ncols() - 1
        } else {
            self.matrix.nrows() - 1
        }
    }

    /// Dimension of vector $f(x)$
    pub fn output_dim(&self) -> usize {
        if self.is_lhs {
            self.matrix.nrows()
        } else {
            self.matrix.ncols()
        }
    }

    /// Apply affine map to a vector
    pub fn map_vec(&self, x: ArrayView1<T>) -> Array1<T> {
        let A = self.get_mul();
        let Ax = if self.is_lhs { A.dot(&x) } else { x.dot(&A) };
        Ax + self.get_shift()
    }

    /// Add equations to the Affine with an augmented matrix
    ///
    /// It's assumes that the `is_lhs`ness of the input is the same as the Affine
    ///
    /// # Panics
    pub fn add_eqns(&mut self, eqns: &Self) {
        if self.is_lhs {
            if eqns.is_lhs {
                self.matrix.append(Axis(0), eqns.matrix.view()).unwrap()
            } else {
                self.matrix.append(Axis(0), eqns.matrix.t()).unwrap()
            }
        } else {
            if eqns.is_lhs {
                self.matrix.append(Axis(1), eqns.matrix.t()).unwrap()
            } else {
                self.matrix.append(Axis(1), eqns.matrix.view()).unwrap()
            }
        };
    }

    /// Get the coefficients of a variable
    pub fn get_var_mut(&mut self, index: usize) -> ArrayViewMut1<T> {
        let axis = if self.is_lhs { Axis(1) } else { Axis(0) };
        self.matrix.index_axis_mut(axis, index)
    }

    pub fn get_coeffs_as_rows(&self) -> ArrayView2<T> {
        if self.is_lhs {
            self.matrix
                .slice_axis(Axis(1), Slice::from(..self.matrix.ncols() - 1))
        } else {
            self.matrix
                .slice_axis(Axis(0), Slice::from(..self.matrix.nrows() - 1))
                .reversed_axes()
        }
    }

    /// # Panics
    pub fn lhs_mul(&self, lhs: &Affine<T>) -> Affine<T> {
        assert!(self.is_lhs);
        assert!(lhs.is_lhs);
        let lhs_matrix = lhs.matrix.view();
        let rhs_matrix = self.matrix.view();
        let mut augmentation: Array2<T> = Array2::zeros((1, rhs_matrix.ncols()));
        augmentation[[0, rhs_matrix.ncols() - 1]] = T::one();
        let aug_rhs_matrix = concatenate!(Axis(0), rhs_matrix, augmentation);
        Self {
            matrix: lhs_matrix.dot(&aug_rhs_matrix),
            is_lhs: true,
        }
    }

    /// # Panics
    pub fn rhs_mul(&self, rhs: &Affine<T>) -> Affine<T> {
        assert!(!self.is_lhs);
        assert!(!rhs.is_lhs);
        let lhs_matrix = self.matrix.view();
        let rhs_matrix = rhs.matrix.view();
        let mut augmentation: Array2<T> = Array2::zeros((lhs_matrix.nrows(), 1));
        augmentation[[self.matrix.nrows() - 1, 0]] = T::one();
        let aug_lhs_matrix = ndarray::concatenate(
            Axis(1),
            &[
                ndarray::ArrayView::from(&lhs_matrix),
                ndarray::ArrayView::from(&augmentation),
            ],
        )
        .unwrap();
        Self {
            matrix: aug_lhs_matrix.dot(&rhs_matrix),
            is_lhs: false,
        }
    }

    pub fn get_mul(&self) -> ArrayView2<T> {
        let slice = if self.is_lhs {
            s![.., ..self.matrix.ncols() - 1]
        } else {
            s![..self.matrix.nrows() - 1, ..]
        };
        self.matrix.slice(slice)
    }

    pub fn get_mul_mut(&mut self) -> ArrayViewMut2<T> {
        let slice = if self.is_lhs {
            s![.., ..self.matrix.ncols() - 1]
        } else {
            s![..self.matrix.nrows() - 1, ..]
        };
        self.matrix.slice_mut(slice)
    }

    pub fn get_shift(&self) -> ArrayView1<T> {
        if self.is_lhs {
            self.matrix.index_axis(Axis(1), self.matrix.ncols() - 1)
        } else {
            self.matrix.index_axis(Axis(0), self.matrix.nrows() - 1)
        }
    }

    pub fn get_shift_mut(&mut self) -> ArrayViewMut1<T> {
        if self.is_lhs {
            self.matrix.index_axis_mut(Axis(1), self.matrix.ncols() - 1)
        } else {
            self.matrix.index_axis_mut(Axis(0), self.matrix.nrows() - 1)
        }
    }
}

impl<T: Float + ndarray::ScalarOperand + std::ops::Mul> Mul<&Self> for Affine<T> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &Self) -> Self {
        let lhs_matrix = self.matrix.view();
        let rhs_matrix = rhs.matrix.view();
        let mut augmentation: Array2<T> = Array2::zeros((lhs_matrix.nrows(), 1));
        augmentation[[self.matrix.nrows() - 1, 0]] = T::one();
        let aug_lhs_matrix = ndarray::concatenate(
            Axis(1),
            &[
                ndarray::ArrayView::from(&lhs_matrix),
                ndarray::ArrayView::from(&augmentation),
            ],
        )
        .unwrap();
        Self {
            matrix: aug_lhs_matrix.dot(&rhs_matrix),
            is_lhs: self.is_lhs,
        }
    }
}

impl<T: Float + ndarray::ScalarOperand + std::ops::Mul> Mul<T> for Affine<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            matrix: &self.get_augmented() * rhs,
            is_lhs: self.is_lhs,
        }
    }
}

impl<T: Float + ndarray::ScalarOperand + std::ops::MulAssign> MulAssign<T> for Affine<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.matrix *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lhs_mul() {
        let twobyfour: Array2<f64> = array![[1., 2., 3., 4.], [5., 6., 7., 8.]];
        let four: Array1<f64> = array![1., 2., 3., 4.];
        let two = array![1., 2.];
        let lhs = Affine::new(twobyfour.clone(), two);
        let rhs = Affine::new(twobyfour.clone().reversed_axes(), four);
        let _out = rhs.lhs_mul(&lhs);
    }

    #[test]
    fn test_rhs_mul() {
        let twobyfour: Array2<f64> = array![[1., 2., 3., 4.], [5., 6., 7., 8.]];
        let four: Array1<f64> = array![1., 2., 3., 4.];
        let two = array![1., 2.];
        let lhs = Affine::new(twobyfour.clone(), four);
        let rhs = Affine::new(twobyfour.clone().reversed_axes(), two);
        let _out = lhs.rhs_mul(&rhs);
    }

    #[test]
    fn test_get_mul_mut() {
        let mut aff: Affine<f64> = Affine::new(Array2::ones([2, 4]), Array1::zeros(4));
        aff.get_mul_mut().assign(&Array2::zeros([2, 4]));
        assert_eq!(aff.get_mul().shape(), [2, 4]);
        assert_eq!(aff.get_mul(), Array2::zeros([2, 4]));
    }

    #[test]
    fn test_get_shift_mut() {
        let mut aff: Affine<f64> = Affine::new(Array2::ones([2, 4]), Array1::zeros(4));
        aff.get_shift_mut().assign(&Array1::ones(4));
        assert_eq!(aff.get_shift().shape(), [4]);
        assert_eq!(aff.get_shift(), Array1::ones(4));
    }

    #[test]
    fn test_get_input_dim() {
        let twobyfour: Array2<f64> = array![[1., 2., 3., 4.], [5., 6., 7., 8.]];
        let four: Array1<f64> = array![1., 2., 3., 4.];
        let two = array![1., 2.];
        let rhs = Affine::new(twobyfour.clone(), four);
        assert_eq!(rhs.input_dim(), 2);
        let lhs = Affine::new(twobyfour.clone(), two);
        assert_eq!(lhs.input_dim(), 4);
    }

    #[test]
    fn test_get_out_dim() {
        let twobyfour: Array2<f64> = array![[1., 2., 3., 4.], [5., 6., 7., 8.]];
        let four: Array1<f64> = array![1., 2., 3., 4.];
        let two = array![1., 2.];
        let rhs = Affine::new(twobyfour.clone(), four);
        assert_eq!(rhs.output_dim(), 4);
        let lhs = Affine::new(twobyfour.clone(), two);
        assert_eq!(lhs.output_dim(), 2);
    }
}
*/

use ndarray::concatenate;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut1;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use ndarray::Slice;
use num::Float;

#[derive(Clone, Debug)]
pub struct Affine<T: Float> {
    pub matrix: Array2<T>,
    pub is_lhs: bool,
}

impl<T: 'static + Float + std::fmt::Display> Affine<T> {
    pub fn new(mul: Array2<T>, shift: Array1<T>) -> Self {
        if mul.ncols() == shift.len() {
            Affine {
                matrix: concatenate!(Axis(0), mul, shift.insert_axis(Axis(0))),
                is_lhs: false,
            }
        } else {
            Affine {
                matrix: concatenate!(Axis(1), mul, shift.insert_axis(Axis(1))),
                is_lhs: true,
            }
        }
    }

    pub fn from_raw(raw: Array2<T>, is_lhs: bool) -> Self {
        Affine {
            matrix: raw,
            is_lhs: is_lhs,
        }
    }

    pub fn input_dim(&self) -> usize {
        if self.is_lhs {
            self.matrix.ncols() - 1
        } else {
            self.matrix.nrows() - 1
        }
    }

    pub fn output_dim(&self) -> usize {
        if self.is_lhs {
            self.matrix.nrows()
        } else {
            self.matrix.ncols()
        }
    }

    pub fn add_eqns(&mut self, eqns: &Affine<T>) {
        if self.is_lhs {
            if !eqns.is_lhs {
                self.matrix.append(Axis(0), eqns.matrix.t()).unwrap()
            } else {
                self.matrix.append(Axis(0), eqns.matrix.view()).unwrap()
            }
        } else {
            if eqns.is_lhs {
                self.matrix.append(Axis(1), eqns.matrix.t()).unwrap()
            } else {
                self.matrix.append(Axis(1), eqns.matrix.view()).unwrap()
            }
        };
    }

    pub fn get_var_mut(&mut self, index: usize) -> ArrayViewMut1<T> {
        let axis = if self.is_lhs { Axis(1) } else { Axis(0) };
        self.matrix.index_axis_mut(axis, index)
    }

    pub fn get_eqn(&self, index: usize) -> ArrayView1<T> {
        let axis = if self.is_lhs { Axis(0) } else { Axis(1) };
        self.matrix.index_axis(axis, index)
    }

    pub fn get_eqn_mut(&mut self, index: usize) -> ArrayViewMut1<T> {
        let axis = if self.is_lhs { Axis(0) } else { Axis(1) };
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

    pub fn lhs_mul(&self, lhs: &Affine<T>) -> Affine<T> {
        assert_eq!(self.is_lhs, true);
        assert_eq!(lhs.is_lhs, true);
        let lhs_matrix = lhs.matrix.view();
        let rhs_matrix = self.matrix.view();
        let mut augmentation: Array2<T> = Array2::zeros((1, rhs_matrix.ncols()));
        augmentation[[0, rhs_matrix.ncols() - 1]] = T::one();
        let aug_rhs_matrix = concatenate!(Axis(0), rhs_matrix, augmentation);
        Affine {
            matrix: lhs_matrix.dot(&aug_rhs_matrix),
            is_lhs: true,
        }
    }

    pub fn rhs_mul(&self, rhs: &Affine<T>) -> Affine<T> {
        assert_eq!(self.is_lhs, false);
        assert_eq!(rhs.is_lhs, false);
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
        Affine {
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

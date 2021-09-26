//! Utility functions
#![allow(non_snake_case)]
use good_lp::solvers::highs::{highs, HighsSolution};
use good_lp::{variable, ResolutionError, Solution, SolverModel};
use good_lp::{Expression, IntoAffineExpression, ProblemVariables, Variable};
use itertools::iproduct;
use ndarray::{s, Axis, Slice};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::SVD;
use num::Float;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

pub fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x).sqrt()
}

/// # Panics
pub fn pinv(x: &Array2<f64>) -> Array2<f64> {
    let (u_opt, sigma, vt_opt) = x.svd(true, true).unwrap();
    let u = u_opt.unwrap();
    let vt = vt_opt.unwrap();
    let sig_diag = &sigma.map(|x| if *x < 1e-10 { 0. } else { 1. / x });
    let mut sig_base = Array2::eye(max(u.nrows(), vt.nrows()));
    sig_base
        .diag_mut()
        .slice_mut(s![..sig_diag.len()])
        .assign(sig_diag);
    let sig = sig_base
        .slice_axis(Axis(0), Slice::from(..vt.nrows()))
        .to_owned();
    let final_sig = sig.slice_axis(Axis(1), Slice::from(..u.nrows()));
    vt.t().dot(&final_sig.dot(&u.t()))
}

/// # Panics
pub fn ensure_spd(A: &Array2<f64>) -> Array2<f64> {
    let B = (A + &A.t()) / 2.;
    let (_, sigma, vt_opt) = A.svd(false, true).unwrap();
    let vt = vt_opt.unwrap();
    let H = vt.t().dot(&sigma).dot(&vt);
    let mut a_hat = (B + H) / 2.;
    // ensure symmetry
    a_hat = (&a_hat + &a_hat.t()) / 2.;
    a_hat
}

pub fn embed_identity(A: &Array2<f64>, dim_opt: Option<usize>) -> Array2<f64> {
    let dim = match dim_opt {
        Some(dim) => dim,
        None => max(A.nrows(), A.ncols()),
    };
    let mut eye = Array2::eye(dim);
    eye.slice_mut(s![..A.nrows(), ..A.ncols()]).assign(A);
    eye
}

/// An linear expression without a constant component
#[derive(Clone)]
pub struct LinearExpression {
    pub coefficients: HashMap<Variable, f64>,
}

impl IntoAffineExpression for LinearExpression {
    type Iter = std::collections::hash_map::IntoIter<Variable, f64>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.coefficients.into_iter()
    }
}

/// Minimizes the expression `c` given the constraint `Ax < b`.
/// # Panics
pub fn solve<'a, I, T: 'a>(
    A: I,
    b: ArrayView1<T>,
    c: ArrayView1<T>,
) -> (Result<HighsSolution, ResolutionError>, Option<f64>)
where
    T: std::convert::Into<f64> + std::clone::Clone + std::marker::Copy + Debug,
    I: IntoIterator<Item = ArrayView1<'a, T>>,
    f64: std::convert::From<T>,
{
    let mut _shh_out;
    let mut _shh_err;
    if !cfg!(test) {
        _shh_out = shh::stdout().unwrap();
        _shh_err = shh::stderr().unwrap();
    }
    let mut problem = ProblemVariables::new();
    let vars = problem.add_vector(variable(), c.len());
    let c_expression = LinearExpression {
        coefficients: vars
            .iter()
            .copied()
            .zip(c.iter().map(|x| f64::from(*x)))
            .collect(),
    };
    let mut unsolved = problem.minimise(c_expression.clone()).using(highs);

    A.into_iter()
        .zip(b.into_iter())
        .for_each(|pair: (ArrayView1<T>, &T)| {
            let (coeffs, ub) = pair;
            let expr = LinearExpression {
                coefficients: vars
                    .iter()
                    .copied()
                    .zip(coeffs.iter().map(|x| f64::from(*x)))
                    .collect(),
            };
            let constr =
                good_lp::constraint::leq(Expression::from_other_affine(expr), f64::from(*ub));
            unsolved.add_constraint(constr);
        });

    let soln = unsolved.solve();
    let fun = soln.as_ref().ok().map(|x| x.eval(c_expression));
    (soln, fun)
}

/// Returns a 2D array of D\[i,j\] = AB\[i,j\] if A\[i,j\] >= 0 and D\[i,j\] = AC\[i,j\] if A\[i,j\] < 0.
///
/// * `A` - The base array of shape `mn`
/// * `B` - The positive array of shape `nk`
/// * `C` - The negative array of shape `nk`
///
/// # Panics
pub fn signed_matmul<T: Float + Sum + Debug>(
    A: &ArrayView2<T>,
    B: &ArrayView2<T>,
    C: &ArrayView2<T>,
) -> Array2<T> {
    assert_eq!(A.ncols(), B.nrows());
    assert_eq!(A.ncols(), C.nrows());
    let mut out = Array2::zeros([A.nrows(), B.ncols()]);
    iproduct!(0..A.nrows(), 0..B.ncols()).for_each(|(i, j)| {
        out[[i, j]] = (0..A.ncols())
            .map(|k| {
                if A[[i, k]] >= T::zero() {
                    A[[i, k]] * B[[k, j]]
                } else {
                    A[[i, k]] * C[[k, j]]
                }
            })
            .sum();
    });
    out
}

/// # Panics
pub fn signed_dot<T: Float + Sum + Debug>(
    A: &ArrayView2<T>,
    B: &ArrayView1<T>,
    C: &ArrayView1<T>,
) -> Array1<T> {
    assert_eq!(A.ncols(), B.len());
    assert_eq!(A.ncols(), C.len());

    let mut out = Array1::zeros(A.nrows());
    (0..A.nrows()).for_each(|i| {
        out[[i]] = (0..A.ncols())
            .map(|k| {
                if A[[i, k]] >= T::zero() {
                    A[[i, k]] * B[[k]]
                } else {
                    A[[i, k]] * C[[k]]
                }
            })
            .sum();
    });
    out
}

pub trait ArenaLike<T> {
    fn new_node(&mut self, data: T) -> usize;
}

impl<T> ArenaLike<T> for Vec<T> {
    fn new_node(&mut self, data: T) -> usize {
        let new_id = self.len();
        self.push(data);
        new_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{array1, array2, bounds1};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_pos_eq_signed_dot(arr1 in array1(3), mut arr2 in array2(3, 3)) {
            arr2 = arr2.map(|x| f64::abs(*x));
            let neg = Array1::ones(3) * -1.;
            let result = signed_dot(&arr2.view(), &arr1.view(), &neg.view());
            prop_assert_eq!(result, &arr2.dot(&arr1));
        }

        #[test]
        fn test_neg_signed_dot(arr1 in array1(3), mut arr2 in array2(3, 3)) {
            arr2 = arr2.map(|x| f64::abs(*x));
            arr2 *= -1.;
            arr2 -= 0.1;
            let pos = Array1::ones(3);
            let result = signed_dot(&arr2.view(), &pos.view(), &arr1.view());
            prop_assert_eq!(result, &arr2.dot(&arr1));
        }

        #[test]
        fn test_signed_dot(bounds in bounds1(3), arr in array2(3, 3)) {
            let lower = signed_dot(&arr.view(), &bounds.lower().view(), &bounds.upper().view());
            let upper = signed_dot(&arr.view(), &bounds.upper().view(), &bounds.lower().view());
            prop_assert!(lower.iter().zip(upper.iter()).all(|(a, b)| a < b));
        }
    }
}

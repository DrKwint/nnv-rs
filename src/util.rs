//! Utility functions
#![allow(non_snake_case)]
use itertools::iproduct;
use ndarray::{s, Axis, Slice};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Scalar;
use ndarray_linalg::SVD;
use ndarray_stats::QuantileExt;
use num::Float;
use rand::Rng;
use std::cmp::max;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::Sum;

pub fn gaussian_logp(x: &ArrayView1<f64>, mu: &ArrayView1<f64>, std: &ArrayView1<f64>) -> f64 {
    let pre_sum: Array1<f64> = ((&(x - mu) / &(std + f64::epsilon())).mapv(f64::square)
        + std.mapv(f64::ln) * (2.)
        + std::f64::consts::TAU.ln())
        * (-0.5);
    pre_sum.sum()
}

pub fn diag_gaussian_accept_reject<R: Rng>(
    x: &ArrayView1<f64>,
    mu: &ArrayView1<f64>,
    sigma: &ArrayView1<f64>,
    rng: &mut R,
) -> bool {
    let likelihood = gaussian_logp(x, mu, sigma).exp();
    let sample: f64 = rng.gen();
    sample < likelihood
    //(0..n_rounds).map(|_| rng.gen()).all(|x: f64| x < likelihood)
}

/// # Panics
pub fn matrix_cond(A: &Array2<f64>, A_inv: &Array2<f64>) -> f64 {
    let (_, sigma, _) = A.svd(false, false).unwrap();
    let (_, inv_sigma, _) = A_inv.svd(false, false).unwrap();
    return sigma.max_skipnan() * inv_sigma.max_skipnan();
}

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
    debug_assert_eq!(A.ncols(), B.nrows());
    debug_assert_eq!(A.ncols(), C.nrows());
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
    debug_assert_eq!(A.ncols(), B.len());
    debug_assert_eq!(A.ncols(), C.len());

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
    fn push_node(&mut self, data: T) -> usize;
}

impl<T> ArenaLike<T> for Vec<T> {
    fn push_node(&mut self, data: T) -> usize {
        let new_id = self.len();
        self.push(data);
        new_id
    }
}

#[derive(Eq, PartialEq)]
pub struct FstOrdTuple<A: Ord, B>(pub (A, B));

impl<A: Ord, B: PartialEq> PartialOrd for FstOrdTuple<A, B> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0 .0.partial_cmp(&other.0 .0)
    }
}

impl<A: Ord, B: Eq> Ord for FstOrdTuple<A, B> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0 .0.cmp(&other.0 .0)
    }
}

/// Calculates the representation step and the actual step that should be taken
///
/// Returns:
///
/// * `representation_step`: The step for the `representation_id`.
/// * `actual_step`: The step used to calculate bounds, stars, etc.
pub fn get_next_step(
    num_steps: Option<usize>,
    step: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    match (num_steps, step) {
        // Steps are not used in the operation
        (None, None) => (None, None),
        // The operation contains only a single step
        (Some(1), None) => (None, Some(0)),
        // If the next step is the last step (step + 1 == num_steps - 1), then we are done with the operation
        (Some(num_steps), Some(step)) if step + 2 == num_steps => (None, Some(step + 1)),
        // If the next step is not the last step, increment
        (Some(num_steps), Some(step)) if step + 2 < num_steps => (Some(step + 1), Some(step + 1)),
        // If we have not yet stepped, step from None to Some
        (Some(_), None) => (Some(0), Some(0)),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{array1, array2, bounds1};
    use float_cmp::assert_approx_eq;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_pos_eq_signed_dot(arr1 in array1(3), mut arr2 in array2(3, 3)) {
            arr2 = arr2.map(|x| f64::abs(*x));
            let neg = Array1::ones(3) * -1.;
            let result = signed_dot(&arr2.view(), &arr1.view(), &neg.view());
            result.into_iter().zip(arr2.dot(&arr1).iter()).for_each(|(x, y)|
                                                                     assert_approx_eq!(f64, x, *y, (1e-12, 16))
                                                                     );
        }

        #[test]
        fn test_neg_signed_dot(arr1 in array1(3), mut arr2 in array2(3, 3)) {
            arr2 = arr2.map(|x| f64::abs(*x));
            arr2 *= -1.;
            arr2 -= 0.1;
            let pos = Array1::ones(3);
            let result = signed_dot(&arr2.view(), &pos.view(), &arr1.view());
            result.into_iter().zip(arr2.dot(&arr1).iter()).for_each(|(x, y)|
                                                                     assert_approx_eq!(f64, x, *y, (1e-12, 16))
                                                                     );
        }

        #[test]
        fn test_signed_dot(bounds in bounds1(3), arr in array2(3, 3)) {
            let lower = signed_dot(&arr.view(), &bounds.lower().view(), &bounds.upper().view());
            let upper = signed_dot(&arr.view(), &bounds.upper().view(), &bounds.lower().view());
            prop_assert!(lower.iter().zip(upper.iter()).all(|(a, b)| a < b));
        }
    }
}

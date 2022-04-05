use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::gaussian::GaussianDistribution;
use crate::lp::solve;
use crate::lp::LinearSolution;
use crate::ndarray_linalg::SVD;
use crate::NNVFloat;
use ndarray::arr1;
use ndarray::array;
use ndarray::concatenate;
use ndarray::Axis;
use ndarray::Ix2;
use ndarray::Slice;
use ndarray::Zip;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};
use ndarray_stats::QuantileExt;
use num::Zero;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::iter;
use std::ops::Deref;
use std::ops::Mul;
use std::ops::MulAssign;
use truncnorm::distributions::MultivariateTruncatedNormal;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Polytope {
    coeffs: Array2<NNVFloat>, // Assume rows correspond to equations and cols are vars, i.e. Ax < b
    rhs: Array1<NNVFloat>,
}

impl Polytope {
    /// # Panics
    pub fn nonempty_new(coeffs: &Array2<NNVFloat>, rhs: &Array1<NNVFloat>) -> Option<Self> {
        let (fcoeffs, frhs): (Vec<ArrayView1<NNVFloat>>, Vec<NNVFloat>) = coeffs
            .rows()
            .into_iter()
            .zip(rhs.iter())
            .filter(|(row, rhs)| row.iter().any(|x| x.abs() > 1e-15) || **rhs < 0.)
            .unzip();
        let fscoeffs: Vec<ArrayView2<NNVFloat>> = fcoeffs
            .into_iter()
            .map(|x| x.insert_axis(Axis(0)))
            .collect();
        if frhs.is_empty() {
            return None;
        }
        let coeffs = concatenate(Axis(0), &fscoeffs).unwrap();
        let rhs = Array1::from_vec(frhs);
        Some(Self::new(coeffs, rhs))
    }

    pub fn new(coeffs: Array2<NNVFloat>, rhs: Array1<NNVFloat>) -> Self {
        debug_assert_eq!(coeffs.nrows(), rhs.len());
        Self { coeffs, rhs }
    }

    pub fn coeffs(&self) -> ArrayView2<NNVFloat> {
        self.coeffs.view()
    }

    pub fn rhs(&self) -> ArrayView1<NNVFloat> {
        self.rhs.view()
    }

    pub fn rhs_mut(&mut self) -> ArrayViewMut1<NNVFloat> {
        self.rhs.view_mut()
    }

    pub fn num_dims(&self) -> usize {
        self.coeffs.ncols()
    }

    pub fn num_constraints(&self) -> usize {
        self.rhs.len()
    }

    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            coeffs: concatenate![Axis(0), self.coeffs, other.coeffs],
            rhs: concatenate![Axis(0), self.rhs, other.rhs],
        }
    }

    pub fn check_redundant(
        &self,
        coeffs: ArrayView1<NNVFloat>,
        rhs: NNVFloat,
        bounds: &Option<Bounds1>,
    ) -> bool {
        let maximize_eqn = &coeffs * -1.;
        let maximize_rhs = rhs + 1.;
        let solved = solve(
            self.coeffs()
                .rows()
                .into_iter()
                .chain(iter::once(maximize_eqn.view())),
            concatenate![Axis(0), self.rhs(), arr1(&[maximize_rhs])].view(),
            maximize_eqn.view(),
            bounds.as_ref(),
        );
        let val: f64 = match solved {
            LinearSolution::Solution(_, val) => val,
            LinearSolution::Infeasible | LinearSolution::Unbounded(_) => return true,
        };
        val > rhs
    }

    /// `check_redundant` is currently disabled
    /// # Panics
    pub fn add_eqn(&mut self, coeffs: ArrayView1<NNVFloat>, rhs: NNVFloat) {
        if coeffs.iter().all(|x| x.abs() < 1e-15) {
            return;
        }
        self.coeffs
            .append(Axis(0), coeffs.insert_axis(Axis(0)).view())
            .unwrap();
        self.rhs.append(Axis(0), array![rhs].view()).unwrap();
    }

    pub fn remove_eqn(&mut self, idx: usize) {
        debug_assert!(idx < self.coeffs().nrows());
        self.coeffs.remove_index(Axis(0), idx);
        self.rhs.remove_index(Axis(0), idx);
    }

    pub fn any_nan(&self) -> bool {
        self.coeffs().iter().any(|x| x.is_nan()) || self.rhs.iter().any(|x| x.is_nan())
    }

    /// # Panics
    pub fn filter_trivial(&mut self) {
        let (coeffs, rhs): (Vec<ArrayView1<NNVFloat>>, Vec<_>) = self
            .coeffs
            .rows()
            .into_iter()
            .zip(self.rhs().iter())
            .filter(|(coeffs, _rhs)| !coeffs.iter().all(|x| *x == 0.))
            .unzip();
        self.coeffs = ndarray::stack(Axis(0), &coeffs).unwrap();
        self.rhs = Array1::from_vec(rhs);
    }

    /// # Panics
    #[must_use]
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

    /// Returns whether a given point is in the set represented by the `Polytope`
    pub fn is_member(&self, point: &ArrayView1<NNVFloat>) -> bool {
        let vals = self.coeffs.dot(point);
        Zip::from(&self.rhs)
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
    }

    /// Remove dimensions from the Polytope that have fixed value.
    ///
    /// # Arguments
    ///
    /// * `x` - An Array with fixed values in each dimension. Any dimensions that aren't fixed should be set to zero.
    /// * `fixed_idxs` - Array that indicates which dimensions are fixed with `true` (because a dim could be fixed at zero)
    ///
    /// # Returns
    ///
    /// None if all the indices are fixed, otherwise a `Self` with reduced dimension
    ///
    /// # Panics
    ///
    /// Only if the underlying struct is malformed
    pub fn reduce_with_values(
        &self,
        x: ArrayView1<NNVFloat>,
        fixed_idxs: ArrayView1<bool>,
    ) -> Option<Self> {
        // Check if every dimension is fixed
        if fixed_idxs.iter().all(|y| *y) {
            return None;
        }

        // Reduce the rhs of each constraint (but don't filter cuz no rows are removed)
        let reduced_rhs: Array1<NNVFloat> = &self.rhs - self.coeffs.dot(&x);

        // Remove the variables that are fixed
        let filtered_coeffs = {
            // Specifically, remove columns
            let filtered_cols: Vec<ArrayView2<NNVFloat>> = self
                .coeffs
                .columns()
                .into_iter()
                .zip(fixed_idxs.iter())
                .filter(|(_item, &is_fixed)| !is_fixed)
                .map(|x| x.0.insert_axis(Axis(1)))
                .collect();
            // We know this unwrap won't error because we haven't changed the number of rows for any column
            concatenate(Axis(1), &filtered_cols).unwrap()
        };

        // Remove trivial constraints (i.e. all zero coeffs on LHS)
        let is_nontrivial: Vec<bool> = filtered_coeffs
            .rows()
            .clone()
            .into_iter()
            .map(|row| row.iter().all(|x| !x.is_zero()))
            .collect();

        // Handle the case where every constraint is trivial
        if !is_nontrivial.iter().any(|y| *y) {
            return None;
        }

        let nontrivial_coeffs: Vec<ArrayView2<NNVFloat>> = filtered_coeffs
            .rows()
            .into_iter()
            .zip(is_nontrivial.iter())
            .filter(|(_, &is_nontrivial)| is_nontrivial)
            .map(|(row, _)| row.insert_axis(Axis(0)))
            .collect();
        let nontrivial_rhs: Vec<NNVFloat> = reduced_rhs
            .into_iter()
            .zip(is_nontrivial.iter())
            .filter(|(_, &is_nontrivial)| is_nontrivial)
            .map(|(val, _)| val)
            .collect();

        let final_coeffs: Array2<NNVFloat> = concatenate(Axis(0), &nontrivial_coeffs).unwrap();
        let final_rhs = Array1::from_vec(nontrivial_rhs);

        Some(Self::new(final_coeffs, final_rhs))
    }

    /// # Panics
    pub fn get_truncnorm_distribution(
        &self,
        mu: ArrayView1<NNVFloat>,
        sigma: ArrayView2<NNVFloat>,
        max_accept_reject_iters: usize,
        stability_eps: NNVFloat,
    ) -> GaussianDistribution {
        // convert T to f64 in inputs
        let mu = mu.mapv(std::convert::Into::into);
        let sigma = sigma.mapv(std::convert::Into::into);
        let mut constraint_coeffs: Array2<f64> = self.coeffs().mapv(std::convert::Into::into);
        let mut ub = self.rhs().mapv(std::convert::Into::into);

        let inv_coeffs = {
            let (u_opt, mut s, vt_opt) = constraint_coeffs.svd(true, true).unwrap();
            let s_max = *s.max().unwrap();
            s /= s_max;
            constraint_coeffs /= s_max;
            ub /= s_max;
            let s_matrix = {
                let mut zeros = Array2::zeros([
                    vt_opt.as_ref().unwrap().shape()[0],
                    u_opt.as_ref().unwrap().shape()[1],
                ]);
                zeros.diag_mut().assign(&s);
                zeros
            };
            vt_opt.unwrap().t().dot(&s_matrix).dot(&u_opt.unwrap().t())
        };

        let sq_constr_sigma = {
            let sigma: Array2<f64> = constraint_coeffs.dot(&sigma.dot(&constraint_coeffs.t()));
            let diag_addn: Array2<f64> =
                Array2::from_diag(&Array1::from_elem(sigma.nrows(), stability_eps));
            sigma + diag_addn
        };
        let sq_ub = ub;

        let sq_constr_ub = &sq_ub - &constraint_coeffs.dot(&mu);
        let sq_constr_lb = Array1::from_elem(sq_constr_ub.len(), f64::NEG_INFINITY);
        let distribution = MultivariateTruncatedNormal::<Ix2>::new(
            mu,
            sq_constr_sigma,
            sq_constr_lb,
            sq_constr_ub,
            max_accept_reject_iters,
        );
        GaussianDistribution::TruncGaussian {
            distribution,
            inv_coeffs,
        }
    }

    /// # Panics
    /// Returns None if the reduced polytope is empty
    pub fn reduce_fixed_inputs(&self, bounds_opt: &Option<Vec<Bounds1>>) -> Option<Self> {
        if bounds_opt.is_none() {
            return Some(self.clone());
        }
        let bounds = bounds_opt
            .as_ref()
            .unwrap()
            .iter()
            .fold(Bounds1::default(0), Bounds1::append);
        let fixed_idxs = bounds.fixed_idxs();
        let fixed_vals = bounds.fixed_vals_or_zeros();

        // update eqns
        self.reduce_with_values(fixed_vals.view(), fixed_idxs.view())
            .map(|mut reduced_poly| {
                reduced_poly.filter_trivial();
                reduced_poly
            })
    }

    /// Check whether the Star set is empty.
    ///
    /// This method assumes that the constraints bound each dimension,
    /// both lower and upper.
    ///
    /// # Panics
    pub fn is_empty<Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        bounds_opt: Option<Bounds1Ref>,
    ) -> bool {
        let c = Array1::ones(self.num_dims());

        let solved = solve(self.coeffs().rows(), self.rhs(), c.view(), bounds_opt);

        !matches!(
            solved,
            LinearSolution::Solution(_, _) | LinearSolution::Unbounded(_)
        )
    }
}

/// Scale by scalar
impl Mul<NNVFloat> for Polytope {
    type Output = Self;

    fn mul(self, rhs: NNVFloat) -> Self {
        Self {
            coeffs: self.coeffs * rhs,
            rhs: self.rhs * rhs,
        }
    }
}

/// Scale by scalar
impl MulAssign<NNVFloat> for Polytope {
    fn mul_assign(&mut self, rhs: NNVFloat) {
        self.coeffs *= rhs;
        self.rhs *= rhs;
    }
}

impl From<Affine2> for Polytope {
    fn from(aff: Affine2) -> Self {
        Self {
            coeffs: aff.basis().to_owned(),
            rhs: aff.shift().to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_util::polytope;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_polytope_get_eqn(ineq in polytope(3, 4)) {
            let eqn = ineq.get_eqn(2);
            let coeffs = eqn.coeffs();
            prop_assert_eq!(&coeffs.shape(), &vec![1, 3])
        }
    }
}

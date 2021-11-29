use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::lp::solve;
use crate::lp::LinearSolution;
use crate::NNVFloat;
use ndarray::arr1;
use ndarray::concatenate;
use ndarray::Axis;
use ndarray::Slice;
use ndarray::Zip;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};
use std::convert::TryFrom;
use std::iter;
use std::ops::Mul;
use std::ops::MulAssign;

#[derive(Clone, Debug)]
pub struct Inequality<T: NNVFloat> {
    coeffs: Array2<T>, // Assume rows correspond to equations and cols are vars, i.e. Ax < b
    rhs: Array1<T>,
    bounds: Bounds1<T>,
}

impl<T: NNVFloat> Inequality<T> {
    pub fn new(coeffs: Array2<T>, rhs: Array1<T>, bounds: Bounds1<T>) -> Self {
        debug_assert_eq!(coeffs.nrows(), rhs.len());
        Self {
            coeffs,
            rhs,
            bounds,
        }
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

    pub fn num_dims(&self) -> usize {
        self.coeffs.ncols()
    }

    pub fn num_constraints(&self) -> usize {
        self.rhs.len()
    }

    pub fn bounds(&self) -> &Bounds1<T> {
        &self.bounds
    }

    pub fn check_redundant(&self, coeffs: ArrayView1<T>, rhs: T) -> bool {
        let neg_one: T = std::convert::From::from(-1.);
        let maximize_eqn = &coeffs * neg_one;
        let maximize_rhs = rhs + T::one();
        let solved = solve(
            self.coeffs()
                .rows()
                .into_iter()
                .chain(iter::once(maximize_eqn.view())),
            concatenate![Axis(0), self.rhs(), arr1(&[maximize_rhs])].view(),
            maximize_eqn.view(),
            &self.bounds,
        );
        let val: f64 = match solved {
            LinearSolution::Solution(_, val) => val,
            LinearSolution::Infeasible | LinearSolution::Unbounded(_) => return true,
        };
        val > rhs.into()
    }

    /// `check_redundant` is currently disabled
    /// # Panics
    pub fn add_eqns(&mut self, eqns: &Self, check_redundant: bool) {
        if check_redundant && false {
            for (c, r) in eqns.coeffs().rows().into_iter().zip(eqns.rhs().into_iter()) {
                if !self.check_redundant(c, *r) {
                    self.coeffs
                        .append(Axis(0), c.insert_axis(Axis(0)))
                        .expect("Failed to add eqn to Inequality");
                    self.rhs
                        .append(Axis(0), arr1(&[*r]).view())
                        .expect("Failed to add eqn to Inequality");
                }
            }
        } else {
            self.coeffs.append(Axis(0), eqns.coeffs()).unwrap();
            self.rhs.append(Axis(0), eqns.rhs()).unwrap();
        }
    }

    pub fn any_nan(&self) -> bool {
        self.coeffs().iter().any(|x| x.is_nan()) || self.rhs.iter().any(|x| x.is_nan())
    }

    /// # Panics
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

    /// # Panics
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
            bounds: self.bounds.get_ith_bounds(idx),
        }
    }

    /// Returns whether a given point is in the set represented by the `Inequality`
    pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
        let vals = self.coeffs.dot(point);
        Zip::from(&self.rhs)
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
    }

    /// Remove dimensions from the inequality that have fixed value.
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
        x: ArrayView1<T>,
        fixed_idxs: ArrayView1<bool>,
    ) -> Option<Self> {
        // Check if every dimension is fixed
        if fixed_idxs.iter().all(|y| *y) {
            return None;
        }

        // Reduce the rhs of each constraint (but don't filter cuz no rows are removed)
        let reduced_rhs: Array1<T> = &self.rhs - self.coeffs.dot(&x);

        // Remove the variables that are fixed
        let filtered_coeffs = {
            // Specifically, remove columns
            let filtered_cols: Vec<ArrayView2<T>> = self
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

        let filtered_bounds_iter = self
            .bounds()
            .bounds_iter()
            .into_iter()
            .zip(&fixed_idxs)
            .filter(|(_, &is_fixed)| !is_fixed)
            .map(|(dim, _drop)| dim);

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

        let nontrivial_coeffs: Vec<ArrayView2<T>> = filtered_coeffs
            .rows()
            .into_iter()
            .zip(is_nontrivial.iter())
            .filter(|(_, &is_nontrivial)| is_nontrivial)
            .map(|(row, _)| row.insert_axis(Axis(0)))
            .collect();
        let nontrivial_rhs: Vec<T> = reduced_rhs
            .into_iter()
            .zip(is_nontrivial.iter())
            .filter(|(_, &is_nontrivial)| is_nontrivial)
            .map(|(val, _)| val)
            .collect();
        let nontrivial_bounds: Vec<ArrayView1<T>> = filtered_bounds_iter
            .zip(is_nontrivial.iter())
            .filter(|(_, &is_nontrivial)| is_nontrivial)
            .map(|(val, _)| val)
            .collect();

        let final_coeffs: Array2<T> = concatenate(Axis(0), &nontrivial_coeffs).unwrap();
        let final_rhs = Array1::from_vec(nontrivial_rhs);
        let final_bounds = Bounds1::new_by_dim(nontrivial_bounds.as_slice());

        Some(Self::new(final_coeffs, final_rhs, final_bounds))
    }
}

/// Scale by scalar
impl<T: NNVFloat> Mul<T> for Inequality<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            coeffs: self.coeffs * rhs,
            rhs: self.rhs * rhs,
            bounds: self.bounds * rhs,
        }
    }
}

/// Scale by scalar
impl<T: NNVFloat> MulAssign<T> for Inequality<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.coeffs *= rhs;
        self.rhs *= rhs;
        self.bounds *= rhs;
    }
}

impl<T: NNVFloat> From<Affine2<T>> for Inequality<T> {
    fn from(aff: Affine2<T>) -> Self {
        Self {
            coeffs: aff.basis().to_owned(),
            rhs: aff.shift().to_owned(),
            bounds: Bounds1::trivial(aff.shift().len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_util::inequality;
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

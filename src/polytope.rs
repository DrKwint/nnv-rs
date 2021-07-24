#![allow(non_snake_case)]
//! Implementation of H-representation polytopes
use crate::inequality::Inequality;
use crate::util::l2_norm;
use crate::util::solve;
use crate::util::LinearExpression;
use good_lp::solvers::highs::highs;
use truncnorm::truncnorm::mv_truncnormal_cdf;

use good_lp::Expression;

use good_lp::ProblemVariables;
use good_lp::ResolutionError;

use good_lp::Variable;
use good_lp::{variable, Solution, SolverModel};
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ScalarOperand;
use ndarray::Zip;

use num::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// H-representation polytope
#[derive(Clone, Debug)]
pub struct Polytope<T: Float> {
    halfspaces: Inequality<T>,
}

impl<T: 'static + Float + Debug> Polytope<T>
where
    T: std::fmt::Display
        + ScalarOperand
        + std::fmt::Debug
        + std::fmt::Display
        + ndarray::ScalarOperand,
    f64: std::convert::From<T>,
{
    pub fn new(constraint_coeffs: Array2<T>, upper_bounds: Array1<T>) -> Self {
        Self {
            halfspaces: Inequality::new(constraint_coeffs, upper_bounds),
        }
    }

    pub fn from_halfspaces(halfspaces: Inequality<T>) -> Self {
        Self { halfspaces }
    }

    pub fn from_input_bounds(lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        // write this to convert bounds into halfspaces
        todo!()
    }

    /// # Panics
    pub fn with_input_bounds(mut self, lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        let lbs = Inequality::new(
            Array2::eye(lower_bounds.len()) * T::from(-1.).unwrap(),
            lower_bounds,
        );
        self.add_constraints(&lbs);
        let ubs = Inequality::new(Array2::eye(upper_bounds.len()), upper_bounds);
        self.add_constraints(&ubs);
        self
    }

    pub fn coeffs(&self) -> ArrayView2<T> {
        self.halfspaces.coeffs()
    }

    pub fn ubs(&self) -> ArrayView1<T> {
        self.halfspaces.rhs()
    }

    pub fn add_constraints(&mut self, constraints: &Inequality<T>) {
        self.halfspaces.add_eqns(constraints)
    }

    pub fn num_constraints(&self) -> usize {
        self.halfspaces.num_constraints()
    }

    /// Check whether the Star set is empty.
    ///
    /// # Panics
    pub fn is_empty(&self) -> bool {
        let mut c = Array1::zeros(self.halfspaces.rhs().len());
        c[[0]] = T::one();

        let solved = solve(
            self.halfspaces.coeffs().rows(),
            self.halfspaces.rhs(),
            c.view(),
        )
        .0;
        !matches!(solved, Ok(_) | Err(ResolutionError::Unbounded))
    }

    /// TODO: doc this
    ///
    /// # Panics
    pub fn gaussian_cdf(
        &self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
    ) -> (f64, f64, f64) {
        let mu = mu.mapv(std::convert::Into::into);
        let sigma = sigma.mapv(std::convert::Into::into);

        let constraint_coeffs = self.coeffs().mapv(std::convert::Into::into);
        let upper_bounds = self.ubs().mapv(std::convert::Into::into);
        let mut sigma_star = constraint_coeffs.dot(&sigma.dot(&constraint_coeffs.t()));
        let pos_def_guarator = Array2::from_diag(&Array1::from_elem(sigma_star.nrows(), 1e-12));
        sigma_star = &sigma_star + pos_def_guarator;
        let ub = &upper_bounds - &constraint_coeffs.dot(&mu);
        let lb = Array1::from_elem(ub.len(), f64::NEG_INFINITY);
        let out = mv_truncnormal_cdf(lb, ub, sigma_star, n, max_iters);
        out
    }
}

/*
/// Source: <https://stanford.edu/class/ee364a/lectures/problems.pdf>
///
/// # Panics
pub fn chebyshev_center(&self) -> (Array1<f64>, f64) {
    let b = self.halfspaces.rhs();
    let mut problem = ProblemVariables::new();
    let r = problem.add_variable();
    let x_c = problem.add_vector(variable(), b.len());
    let mut unsolved = problem.maximise(r).using(highs);

    self.halfspaces
        .get_coeffs_as_rows()
        .rows()
        .into_iter()
        .zip(b.into_iter())
        .for_each(|pair: (ArrayView1<T>, &T)| {
            let (coeffs, ub) = pair;
            let coeffs = coeffs.map(|x| f64::from(*x));
            let l2_norm_val = l2_norm(coeffs.view());
            let mut expr_map: HashMap<Variable, f64> =
                x_c.iter().copied().zip(coeffs).collect();
            expr_map.insert(r, l2_norm_val);
            let expr = LinearExpression {
                coefficients: expr_map,
            };
            let constr =
                good_lp::constraint::leq(Expression::from_other_affine(expr), f64::from(*ub));
            unsolved.add_constraint(constr);
        });
    let soln = unsolved.solve().unwrap();
    let x_c: Array1<f64> = x_c.into_iter().map(|x| soln.value(x)).collect();
    (x_c, soln.value(r))
}

pub fn reduce_fixed_inputs(&self) -> Self {
    if let Some((halfspaces, (lbs, ubs))) = self
        .halfspaces
        .as_ref()
        .zip(self.lower_bounds.as_ref().zip(self.upper_bounds.as_ref()))
    {
        let fixed_idxs = Zip::from(lbs).and(ubs).map_collect(|&lb, &ub| !(lb == ub));
        let fixed_vals =
            Zip::from(lbs)
                .and(ubs)
                .map_collect(|&lb, &ub| if lb == ub { lb } else { T::zero() });

        // update eqns
        let new_halfspaces =
            halfspaces.reduce_with_values(fixed_vals.view(), fixed_idxs.view());

        // update bounds
        let new_lbs: Array1<T> = lbs
            .into_iter()
            .zip(&fixed_idxs)
            .filter(|(_lb, &is_fix)| is_fix)
            .map(|(&lb, _is_fix)| lb)
            .collect();
        let new_ubs: Array1<T> = ubs
            .into_iter()
            .zip(&fixed_idxs)
            .filter(|(_ub, &is_fix)| is_fix)
            .map(|(&ub, _is_fix)| ub)
            .collect();

        Self {
            halfspaces: new_halfspaces,
        }
    } else {
        self.clone()
    }
}

pub fn coeffs(&self) -> Option<ArrayView2<T>> {
    self.halfspaces.as_ref().map(crate::affine::Affine::get_mul)
}

pub fn get_coeffs_as_rows(&self) -> Option<ArrayView2<T>> {
    self.halfspaces
        .as_ref()
        .map(crate::affine::Affine::get_coeffs_as_rows)
}

/// # Panics
pub fn eqn_upper_bounds(&self) -> ArrayView1<T> {
    self.halfspaces.as_ref().unwrap().get_shift()
}

/// # Panics
pub fn num_constraints(&self) -> usize {
    self.halfspaces.as_ref().unwrap().output_dim()
}

pub fn add_constraints(&mut self, constraints: &Affine<T>) {
    match self.halfspaces.as_mut() {
        Some(x) => x.add_eqns(constraints),
        None => self.halfspaces = Some(constraints.clone()),
    }
}

/// # Panics
pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
    if self.coeffs().is_none() {
        return true;
    }
    let vals = point.dot(&self.coeffs().unwrap());
    Zip::from(self.eqn_upper_bounds())
        .and(&vals)
        .fold(true, |acc, ub, v| acc && (v <= ub))
}

// <https://mathoverflow.net/questions/9854/uniformly-sampling-from-convex-polytopes>
// <https://arxiv.org/pdf/2007.01578.pdf>
//pub fn uniform_sample()
*/

/*
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_empty() {
        let coeffs = array![[0., 1.], [1., 0.]];
        let ubs = array![0., 0.];
        let pt = Polytope::new(coeffs, ubs);
        assert!(!pt.is_empty());
        let coeffs = array![[0., 0., 1.], [0., 0., -1.]];
        let ubs = array![0., -1.];
        let pt = Polytope::new(coeffs, ubs);
        assert!(pt.is_empty());
    }

    #[test]
    fn test_member() {
        let coeffs = array![[1., 0.], [0., 1.], [1., -1.]].reversed_axes();
        let ubs = array![0., 0., 0.];
        let pt = Polytope::new(coeffs, ubs);
        let _points = vec![array![0., 0.], array![1., 1.], array![0., 1.]];
        todo!()
    }
}
*/

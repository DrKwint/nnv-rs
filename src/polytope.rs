#![allow(non_snake_case, clippy::similar_names)]
//! Implementation of H-representation polytopes
use crate::bounds::Bounds1;
use crate::inequality::Inequality;
use crate::rand::distributions::Distribution;
use rand::Rng;

use crate::util::embed_identity;
use crate::util::l2_norm;
use crate::util::solve;
use crate::util::LinearExpression;
use good_lp::solvers::highs::highs;
use ndarray::Slice;
use ndarray::{concatenate, s, Array};
use ndarray_linalg::solve::Inverse;
use truncnorm::distributions::MultivariateTruncatedNormal;
use truncnorm::truncnorm::mv_truncnormal_cdf;
use truncnorm::truncnorm::mv_truncnormal_rand;

use good_lp::Expression;

use good_lp::ProblemVariables;
use good_lp::ResolutionError;

use good_lp::Variable;
use good_lp::{variable, Solution, SolverModel};
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Ix1;
use ndarray::ScalarOperand;
use ndarray::Zip;
use ndarray::{array, Array1, Axis};

use num::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// H-representation polytope
#[derive(Clone, Debug)]
pub struct Polytope<T: Float> {
    halfspaces: Inequality<T>,
}

impl<T: 'static> Polytope<T>
where
    T: Float + ScalarOperand + From<f64>,
    f64: From<T>,
{
    pub fn new(constraint_coeffs: Array2<T>, upper_bounds: Array1<T>) -> Self {
        Self {
            halfspaces: Inequality::new(constraint_coeffs, upper_bounds),
        }
    }

    pub fn from_halfspaces(halfspaces: Inequality<T>) -> Self {
        Self { halfspaces }
    }

    /// # Panics
    pub fn with_input_bounds(mut self, input_bounds: Bounds1<T>) -> Self {
        let item = Self::from(input_bounds);
        self.halfspaces.add_eqns(&item.halfspaces);
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

    pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
        let vals = self.coeffs().dot(point);
        Zip::from(self.ubs())
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
    }

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
        mv_truncnormal_cdf(lb, ub, sigma_star, n, max_iters)
    }

    /// # Panics
    #[allow(clippy::too_many_lines)]
    pub fn gaussian_sample<R: Rng>(
        &self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
    ) -> Vec<(Array1<f64>, f64)> {
        // convert T to f64 in inputs
        let mu = mu.mapv(std::convert::Into::into);
        let sigma = sigma.mapv(std::convert::Into::into);

        // sample unfixed dimensions
        let mut constraint_coeffs: Array2<f64> = self.coeffs().mapv(|x| x.into());
        // normalise each equation
        let row_norms: Array1<f64> = constraint_coeffs
            .rows()
            .into_iter()
            .map(|row| row.mapv(|x| x * x).sum().sqrt())
            .collect();
        constraint_coeffs = (&constraint_coeffs.t() / &row_norms).reversed_axes();
        let ub = self.ubs().mapv(std::convert::Into::into) / row_norms;

        // embed constraint coeffs in an identity matrix
        let sq_coeffs = embed_identity(&constraint_coeffs, None).reversed_axes();
        // if there are more constraints than variables, add dummy variables
        let sq_sigma = embed_identity(&sigma, Some(sq_coeffs.nrows()));
        let sq_constr_sigma = {
            let sigma: Array2<f64> = sq_coeffs.dot(&sq_sigma.dot(&sq_coeffs.t()));
            let diag_addn = Array2::from_diag(&Array1::from_elem(sigma.nrows(), 1e-12));
            sigma + diag_addn
        };
        let mut sq_ub = Array::from_elem(sq_coeffs.nrows(), f64::INFINITY);
        sq_ub.slice_mut(s![..ub.len()]).assign(&ub);

        let extended_reduced_mu = if sq_coeffs.nrows() == mu.len() {
            mu.clone()
        } else {
            let mut e_r_mu = Array1::zeros(sq_coeffs.nrows());
            e_r_mu.slice_mut(s![..mu.len()]).assign(&mu);
            e_r_mu
        };

        let sq_constr_ub = &sq_ub - &sq_coeffs.dot(&extended_reduced_mu);
        let sq_constr_lb = Array1::from_elem(sq_constr_ub.len(), f64::NEG_INFINITY);
        let (centered_samples, logp) = if sq_constr_sigma.len() == 1 {
            let sample = MultivariateTruncatedNormal::<Ix1>::new(
                array![0.],
                sq_constr_sigma.index_axis(Axis(0), 0).to_owned(),
                sq_constr_lb,
                sq_constr_ub,
                max_iters,
            )
            .sample(rng);
            (sample.insert_axis(Axis(1)), array![1.])
        } else {
            mv_truncnormal_rand(sq_constr_lb, sq_constr_ub, sq_constr_sigma, n, max_iters)
        };
        let inv_constraint_coeffs = &sq_coeffs.inv().unwrap();
        let mut samples = inv_constraint_coeffs
            .dot(&centered_samples.t())
            .reversed_axes();
        samples = samples
            .slice_axis(Axis(1), Slice::from(0..sigma.nrows()))
            .to_owned();
        let mut filtered_samples: Vec<(Array1<f64>, f64)> = samples
            .rows()
            .into_iter()
            .zip(logp)
            .map(|(x, logp)| (x.to_owned() + &mu, logp))
            .filter(|(x, _logp)| self.is_member(&x.mapv(|v| v.into()).view()))
            .collect();
        if filtered_samples.is_empty() {
            filtered_samples = if let Some((x_c, _r)) = self.chebyshev_center() {
                vec![(x_c, 0.43)]
            } else {
                vec![(mu, 0.43)]
            };
        };
        filtered_samples
    }

    /// Source: <https://stanford.edu/class/ee364a/lectures/problems.pdf>
    pub fn chebyshev_center(&self) -> Option<(Array1<f64>, f64)> {
        let b = self.halfspaces.rhs();
        let mut problem = ProblemVariables::new();
        let r = problem.add_variable();
        let x_c = problem.add_vector(variable(), b.len());
        let mut unsolved = problem.maximise(r).using(highs);

        self.halfspaces
            .coeffs()
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
        if let Ok(soln) = unsolved.solve() {
            let x_c: Array1<f64> = x_c.into_iter().map(|x| soln.value(x)).collect();
            Some((x_c, soln.value(r)))
        } else {
            None
        }
    }

    pub fn reduce_fixed_inputs(
        &self,
        lbs: &ArrayView1<T>,
        ubs: &ArrayView1<T>,
    ) -> (Self, (Array1<T>, Array1<T>)) {
        let fixed_idxs = Zip::from(lbs).and(ubs).map_collect(|&lb, &ub| !(lb == ub));
        let fixed_vals = Zip::from(lbs)
            .and(ubs)
            .map_collect(|&lb, &ub| if lb == ub { lb } else { T::zero() });

        // update eqns
        let new_halfspaces = self
            .halfspaces
            .reduce_with_values(fixed_vals.view(), fixed_idxs.view());

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

        let reduced_poly = Self {
            halfspaces: new_halfspaces,
        };
        (reduced_poly, (new_lbs, new_ubs))
    }
}

impl<T: 'static> Polytope<T>
where
    T: Float + ScalarOperand + From<f64> + Debug,
    f64: From<T>,
{
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
}

// Allow a technically fallible from because we're matching array shapes in the fn body
#[allow(clippy::fallible_impl_from)]
impl<T: Float + ScalarOperand> From<Bounds1<T>> for Polytope<T> {
    fn from(item: Bounds1<T>) -> Self {
        let coeffs = concatenate(
            Axis(0),
            &[
                (Array2::eye(item.ndim()) * T::neg(T::one())).view(),
                Array2::eye(item.ndim()).view(),
            ],
        )
        .unwrap();
        let rhs = concatenate(Axis(0), &[item.lower(), item.upper()]).unwrap();
        let halfspaces = Inequality::new(coeffs, rhs);
        Self { halfspaces }
    }
}

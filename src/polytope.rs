#![allow(non_snake_case, clippy::similar_names)]
//! Implementation of H-representation polytopes
use crate::bounds::Bounds1;
use crate::gaussian::GaussianDistribution;
use crate::inequality::Inequality;
use crate::lp::solve;
use crate::lp::LinearSolution;
use crate::util;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Ix2;
use std::fmt::Debug;
use truncnorm::distributions::MultivariateTruncatedNormal;

/// H-representation polytope
#[derive(Clone, Debug)]
pub struct Polytope<T: NNVFloat> {
    halfspaces: Inequality<T>,
}

impl<T: NNVFloat> Polytope<T> {
    pub fn new(constraint_coeffs: Array2<T>, upper_bounds: Array1<T>, bounds: Bounds1<T>) -> Self {
        Self {
            halfspaces: Inequality::new(constraint_coeffs, upper_bounds, bounds),
        }
    }

    pub fn from_halfspaces(halfspaces: Inequality<T>) -> Self {
        Self { halfspaces }
    }

    /// # Panics
    pub fn with_input_bounds(mut self, input_bounds: Bounds1<T>) -> Self {
        let item = Self::from(input_bounds);
        self.halfspaces.add_eqns(&item.halfspaces, false);
        self
    }

    pub fn coeffs(&self) -> ArrayView2<T> {
        self.halfspaces.coeffs()
    }

    pub fn num_vars(&self) -> usize {
        self.halfspaces.coeffs().ncols()
    }

    pub fn ubs(&self) -> ArrayView1<T> {
        self.halfspaces.rhs()
    }

    pub fn get_bounds(&self) -> &Bounds1<T> {
        self.halfspaces.bounds()
    }

    pub fn add_constraints(&mut self, constraints: &Inequality<T>, check_redundant: bool) {
        self.halfspaces.add_eqns(constraints, check_redundant);
    }

    pub fn num_constraints(&self) -> usize {
        self.halfspaces.num_constraints()
    }

    pub fn any_nan(&self) -> bool {
        self.halfspaces.any_nan()
    }

    pub fn filter_trivial(&mut self) {
        self.halfspaces.filter_trivial();
    }

    pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
        self.halfspaces.is_member(point)
    }

    pub fn get_truncnorm_distribution(
        &self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        max_accept_reject_iters: usize,
        stability_eps: T,
    ) -> GaussianDistribution<T> {
        // convert T to f64 in inputs
        let mu = mu.mapv(std::convert::Into::into);
        let sigma = sigma.mapv(std::convert::Into::into);
        let mut constraint_coeffs: Array2<f64> = self.coeffs().mapv(std::convert::Into::into);
        let mut ub = self.ubs().mapv(std::convert::Into::into);

        // normalise each constraint equation
        let row_norms: Array1<f64> = constraint_coeffs
            .rows()
            .into_iter()
            .map(|row| row.mapv(|x| x * x).sum().sqrt())
            .collect();
        debug_assert!(
            !row_norms.iter().any(|x| *x == 0.),
            "{:?}",
            constraint_coeffs.rows().into_iter().collect::<Vec<_>>()
        );
        constraint_coeffs = (&constraint_coeffs.t() / &row_norms).reversed_axes();
        ub = ub / row_norms;

        let sq_constr_sigma = {
            let sigma: Array2<f64> = constraint_coeffs.dot(&sigma.dot(&constraint_coeffs.t()));
            let diag_addn: Array2<f64> =
                Array2::from_diag(&Array1::from_elem(sigma.nrows(), stability_eps.into()));
            //println!("sigma cond before stability: {:?}", util::matrix_cond(&sigma, &sigma.inv().unwrap()));
            //println!("sigma cond after stability: {:?}", util::matrix_cond(&(&sigma + &diag_addn), &(&sigma + &diag_addn).inv().unwrap()));
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
        let inv_coeffs: Array2<T> = util::pinv(&constraint_coeffs).mapv(Into::into);
        // println!("coeffs cond: {:?}", util::matrix_cond(&constraint_coeffs, &inv_coeffs.mapv(|x| x.into())));
        GaussianDistribution::TruncGaussian {
            distribution,
            inv_coeffs,
        }
    }

    /* Commented out while we're working on building the lp features
    /// Source: <https://stanford.edu/class/ee364a/lectures/problems.pdf>
    pub fn chebyshev_center(&self) -> Option<(Array1<f64>, f64)> {
        let b = self.halfspaces.rhs();
        let mut problem = ProblemVariables::new();
        let r = problem.add_variable();
        let x_c = problem.add_vector(variable(), b.len());
        let mut unsolved = problem.maximise(r).using(coin_cbc);

        self.halfspaces
            .coeffs()
            .rows()
            .into_iter()
            .zip(b.into_iter())
            .for_each(|pair: (ArrayView1<T>, &T)| {
                let (coeffs, ub) = pair;
                let coeffs = coeffs.map(|x| (*x).into());
                let l2_norm_val = l2_norm(coeffs.view());
                let mut expr_map: Vec<(Variable, f64)> = x_c.iter().copied().zip(coeffs).collect();
                expr_map.push((r, l2_norm_val));
                let expr = LinearExpression {
                    coefficients: expr_map,
                };
                let constr =
                    good_lp::constraint::leq(Expression::from_other_affine(expr), (*ub).into());
                unsolved.add_constraint(constr);
            });
        if let Ok(soln) = unsolved.solve() {
            let x_c: Array1<f64> = x_c.into_iter().map(|x| soln.value(x)).collect();
            Some((x_c, soln.value(r)))
        } else {
            None
        }
    }
    */

    /// Returns None if the reduced polytope is empty
    pub fn reduce_fixed_inputs(&self) -> Option<Self> {
        let bounds = self.get_bounds();
        let fixed_idxs = bounds.fixed_idxs();
        let fixed_vals = bounds.fixed_vals_or_zeros();

        // update eqns
        self.halfspaces
            .reduce_with_values(fixed_vals.view(), fixed_idxs.view())
            .map(|halfspaces| {
                let mut reduced_poly = Self { halfspaces };
                reduced_poly.filter_trivial();
                reduced_poly
            })
    }
}

impl<T: NNVFloat> Polytope<T> {
    /// Check whether the Star set is empty.
    ///
    /// This method assumes that the constraints bound each dimension,
    /// both lower and upper.
    ///
    /// # Panics
    pub fn is_empty(&self) -> bool {
        let c = Array1::ones(self.num_vars());

        let solved = solve(
            self.halfspaces.coeffs().rows(),
            self.halfspaces.rhs(),
            c.view(),
            self.get_bounds(),
        );

        !matches!(solved, LinearSolution::Solution(_, _))
    }
}

// Allow a technically fallible from because we're matching array shapes in the fn body
#[allow(clippy::fallible_impl_from)]
impl<T: NNVFloat> From<Bounds1<T>> for Polytope<T> {
    fn from(item: Bounds1<T>) -> Self {
        let coeffs = Array2::zeros((1, item.ndim()));
        let rhs = Array1::zeros(1);
        let halfspaces = Inequality::new(coeffs, rhs, item);
        Self { halfspaces }
    }
}

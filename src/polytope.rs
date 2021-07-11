#![allow(non_snake_case)]
use crate::affine::Affine;
use crate::util::l2_norm;
use crate::util::solve;
use crate::util::LinearExpression;
use good_lp::solvers::highs::highs;

use good_lp::Expression;

use good_lp::ProblemVariables;
use good_lp::ResolutionError;

use good_lp::Variable;
use good_lp::{variable, Solution, SolverModel};
use ndarray::concatenate;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::ScalarOperand;
use ndarray::Zip;

use num::Float;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Polytope<T: Float> {
    halfspaces: Option<Affine<T>>,
    lower_bounds: Option<Array1<T>>,
    upper_bounds: Option<Array1<T>>,
}

impl<T: 'static + Float + Debug> Polytope<T>
where
    T: std::convert::Into<f64>,
    T: std::fmt::Display,
    T: ScalarOperand,
    f64: std::convert::From<T>,
{
    pub fn new(constraint_coeffs: Array2<T>, upper_bounds: Array1<T>) -> Self {
        Polytope {
            halfspaces: Some(Affine::new(constraint_coeffs, upper_bounds)),
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn from_affine(halfspaces: Affine<T>) -> Self {
        Polytope {
            halfspaces: Some(halfspaces),
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn from_input_bounds(lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        Self {
            halfspaces: None,
            lower_bounds: Some(lower_bounds),
            upper_bounds: Some(upper_bounds),
        }
        //poly.with_input_bounds(lower_bounds, upper_bounds)
    }

    pub fn with_input_bounds(mut self, lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        self.lower_bounds = Some(lower_bounds.clone());
        self.upper_bounds = Some(upper_bounds.clone());
        let is_lhs = self.halfspaces.as_ref().unwrap().is_lhs;
        let axis = if is_lhs { Axis(1) } else { Axis(0) };
        let lbs = concatenate(
            axis,
            &[
                (Array2::eye(lower_bounds.len()) * T::from(-1.).unwrap()).view(),
                lower_bounds.insert_axis(axis).view(),
            ],
        )
        .unwrap();
        self.add_constraints(&Affine::from_raw(lbs, is_lhs));
        let ubs = concatenate(
            axis,
            &[
                Array2::eye(upper_bounds.len()).view(),
                upper_bounds.insert_axis(axis).view(),
            ],
        )
        .unwrap();
        self.add_constraints(&Affine::from_raw(ubs, is_lhs));
        self
    }

    pub fn get_input_bounds(&self) -> Option<(ArrayView1<T>, ArrayView1<T>)> {
        self.lower_bounds
            .as_ref()
            .map(|x| x.view())
            .zip(self.upper_bounds.as_ref().map(|x| x.view()))
    }

    pub fn get_input_lower_bound(&self) -> Option<ArrayView1<T>> {
        self.lower_bounds.as_ref().map(|x| x.view())
    }

    pub fn get_input_upper_bound(&self) -> Option<ArrayView1<T>> {
        self.upper_bounds.as_ref().map(|x| x.view())
    }

    pub fn in_bounds(&self, x: &ArrayView1<T>) -> bool {
        let lbs = self.lower_bounds.as_ref().unwrap();
        let ubs = self.upper_bounds.as_ref().unwrap();
        let fixed_idxs = Zip::from(x)
            .and(lbs)
            .and(ubs)
            .map_collect(|&v, &lb, &ub| v >= lb && v <= ub);
        fixed_idxs.iter().all(|&x| x)
    }

    pub fn reduce_fixed_inputs(&self) -> Self {
        let lbs = self.lower_bounds.as_ref().unwrap();
        let ubs = self.upper_bounds.as_ref().unwrap();
        let fixed_idxs = Zip::from(lbs)
            .and(ubs)
            .map_collect(|&lb, &ub| !(lb == ub));
        let fixed = Zip::from(lbs)
            .and(ubs)
            .map_collect(|&lb, &ub| if lb == ub { lb } else { T::zero() });

        // update eqns
        if self.halfspaces.is_none() {
            return self.clone();
        }
        let ub_reduction = if self.halfspaces.as_ref().unwrap().is_lhs {
            self.coeffs().unwrap().dot(&fixed)
        } else {
            fixed.dot(&self.coeffs().unwrap())
        };
        let new_eqn_ubs = &self.eqn_upper_bounds() - ub_reduction;
        let vars = self.get_coeffs_as_rows().unwrap();
        let new_eqns_vec: Vec<ArrayView2<T>> = vars
            .columns()
            .into_iter()
            .zip(&fixed_idxs)
            .filter(|(_, &fixed)| fixed)
            .map(|(var, _)| var.insert_axis(Axis(0)))
            .collect();
        let new_eqns = concatenate(Axis(0), new_eqns_vec.as_slice()).unwrap();

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
            .filter(|(_lb, &is_fix)| is_fix)
            .map(|(&lb, _is_fix)| lb)
            .collect();

        Polytope {
            halfspaces: Some(Affine::new(new_eqns, new_eqn_ubs)),
            lower_bounds: Some(new_lbs),
            upper_bounds: Some(new_ubs),
        }
    }

    pub fn coeffs(&self) -> Option<ArrayView2<T>> {
        self.halfspaces.as_ref().map(|x| x.get_mul())
    }

    pub fn get_coeffs_as_rows(&self) -> Option<ArrayView2<T>> {
        self.halfspaces.as_ref().map(|x| x.get_coeffs_as_rows())
    }

    pub fn eqn_upper_bounds(&self) -> ArrayView1<T> {
        self.halfspaces.as_ref().unwrap().get_shift()
    }

    pub fn num_constraints(&self) -> usize {
        self.halfspaces.as_ref().unwrap().output_dim()
    }

    pub fn add_constraints(&mut self, constraints: &Affine<T>) {
        match self.halfspaces.as_mut() {
            Some(x) => x.add_eqns(constraints),
            None => self.halfspaces = Some(constraints.clone()),
        }
    }

    pub fn affine_map(&self, affine: &Affine<T>) -> Affine<T> {
        let halfspaces = self.halfspaces.as_ref().unwrap();
        assert_eq!(halfspaces.is_lhs, affine.is_lhs);
        if halfspaces.is_lhs {
            halfspaces.lhs_mul(affine)
        } else {
            halfspaces.rhs_mul(affine)
        }
    }

    pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
        if self.coeffs().is_none() {
            return true;
        }
        let vals = point.dot(&self.coeffs().unwrap());
        Zip::from(self.eqn_upper_bounds())
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
    }

    /// Check whether the Star set is empty.
    pub fn is_empty(&self) -> bool {
        let mut c = Array1::zeros(self.eqn_upper_bounds().len());
        c[[0]] = T::one();

        let solved = solve(
            self.halfspaces
                .as_ref()
                .unwrap()
                .get_coeffs_as_rows()
                .rows(),
            self.eqn_upper_bounds(),
            c.view(),
            self.lower_bounds.as_ref().map(|x| x.view()),
            self.upper_bounds.as_ref().map(|x| x.view()),
        )
        .0;
        !matches!(solved, Ok(_) | Err(ResolutionError::Unbounded))
    }

    /// Source: <https://stanford.edu/class/ee364a/lectures/problems.pdf>
    pub fn chebyshev_center(&self) -> (Array1<f64>, f64) {
        let b = self.eqn_upper_bounds();
        let mut problem = ProblemVariables::new();
        let r = problem.add_variable();
        let x_c = if self.lower_bounds.is_some() {
            let lowers = self.lower_bounds.as_ref().unwrap();
            let uppers = self.upper_bounds.as_ref().unwrap();
            lowers
                .into_iter()
                .zip(uppers)
                .map(|bounds| {
                    problem.add(
                        variable()
                            .min(f64::from(*bounds.0))
                            .max(f64::from(*bounds.1)),
                    )
                })
                .collect()
        } else {
            problem.add_vector(variable(), b.len())
        };
        let mut unsolved = problem.maximise(r).using(highs);

        self.halfspaces
            .as_ref()
            .unwrap()
            .get_coeffs_as_rows()
            .rows()
            .into_iter()
            .zip(b.into_iter())
            .for_each(|pair: (ArrayView1<T>, &T)| {
                let (coeffs, ub) = pair;
                let coeffs = coeffs.map(|x| f64::from(*x));
                let l2_norm_val = l2_norm(coeffs.view());
                let mut expr_map: HashMap<Variable, f64> =
                    x_c.iter().cloned().zip(coeffs).collect();
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

    // <https://mathoverflow.net/questions/9854/uniformly-sampling-from-convex-polytopes>
    // <https://arxiv.org/pdf/2007.01578.pdf>
    //pub fn uniform_sample()
}

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
    }
}

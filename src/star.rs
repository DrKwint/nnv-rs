extern crate truncnorm;

use crate::affine::Affine;
use crate::polytope::Polytope;
use crate::util::pinv;
use crate::util::solve;
use good_lp::ResolutionError;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use num::Float;
use std::fmt::Debug;
use truncnorm::mv_truncnormal_cdf;
use truncnorm::mv_truncnormal_rand;

/// Representation of a set acted on by a deep neural network (DNN)
///
/// Star sets are defined by a 1) constraint coefficient matrix, 2) upper
/// bound vector, 3) basis matrix, 4) center vector. (1) and (2) define a
/// polyhedron and (3) and (4) define an affine transformation of that
/// polyhedron.
///
/// Each Star set represents two sets implicitly: an input set and a
/// representation set. The input set is defined in the input space of the deep
/// neural network of interest. It's a polyhedron defined by the Star's
/// constraints (coefficient matrix and upper bound vector). The representation
/// set is defined in a latent or output space of the DNN. It is calculated by
/// applying the affine transformation defined by the Star's basis and center
/// to the input set polyhedron.
///
/// Shapes
/// basis - [input_dim, repr_dim]
/// center - [repr_dim]
/// constraints - [num_constraints, input_dim]
/// upper_bounds - [num_constraints]
#[derive(Clone, Debug)]
pub struct Star<T: Float> {
    /// `representation` is the concatenation of [basis center] (where center is a column vector) and captures information about the transformed set
    representation: Affine<T>,
    /// `constraints` is the concatenation of [coeffs upper_bounds] and is a representation of the input polyhedron
    constraints: Option<Polytope<T>>,
    input_lower_bounds: Option<Array1<T>>,
    input_upper_bounds: Option<Array1<T>>,
}

impl<T: Float> Star<T>
where
    T: std::convert::From<f64>,
    T: std::convert::Into<f64>,
    T: ndarray::ScalarOperand,
    T: std::fmt::Display,
    T: std::fmt::Debug,
    f64: std::convert::From<T>,
{
    /// Create a new Star with given dimension.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    pub fn default(dim: usize) -> Self {
        Star {
            representation: Affine::new(Array2::eye(dim), Array1::zeros(dim)),
            constraints: None,
            input_lower_bounds: None,
            input_upper_bounds: None,
        }
    }

    /// Create a new Star with given basis vector and center.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    pub fn new(basis: Array2<T>, center: Array1<T>) -> Self {
        Star {
            representation: Affine::new(basis, center),
            constraints: None,
            input_lower_bounds: None,
            input_upper_bounds: None,
        }
    }

    pub fn with_input_bounds(mut self, lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        self.input_lower_bounds = Some(lower_bounds);
        self.input_upper_bounds = Some(upper_bounds);
        self
    }

    pub fn input_space_polytope(&self) -> Option<&Polytope<T>> {
        self.constraints.as_ref()
    }

    /// Get the dimension of the input space
    pub fn input_space_dim(&self) -> usize {
        self.representation.input_dim()
    }

    /// Get the dimension of the representation space
    pub fn representation_space_dim(&self) -> usize {
        self.representation.output_dim()
    }

    pub fn num_constraints(&self) -> usize {
        match &self.constraints {
            Some(polytope) => polytope.num_constraints(),
            None => 0,
        }
    }

    /// `basis` is the matrix used to transform from the input space to the representation space.
    pub fn basis(&self) -> ArrayView2<T> {
        self.representation.get_mul()
    }

    /// `center` is the column vector which shifts all points in the input and representation polyhedra from the origin.
    pub fn center(&self) -> ArrayView1<T> {
        self.representation.get_shift()
    }

    pub fn constraint_coeffs(&self) -> Option<ArrayView2<T>> {
        if let Some(constrs) = &self.constraints {
            Some(constrs.coeffs())
        } else {
            None
        }
    }

    pub fn constraint_upper_bounds(&self) -> Option<ArrayView1<T>> {
        if let Some(constrs) = &self.constraints {
            Some(constrs.upper_bounds())
        } else {
            None
        }
    }

    /// Add constraints to restrict the input set. Each row represents a
    /// constraint and the last column represents the upper bounds.
    pub fn add_constraints(mut self, new_constraints: &Affine<T>) -> Self {
        // assert_eq!(self.representation.is_lhs, new_constraints.is_lhs);
        if let Some(ref mut constrs) = self.constraints {
            constrs.add_constraints(new_constraints);
        } else {
            self.constraints = Some(Polytope::from_affine(new_constraints.clone()));
        }
        self
    }

    /// Apply an affine transformation to the representation polyhedron
    pub fn affine_map(&self, affine: &Affine<T>) -> Self {
        let new_repr = self.representation.rhs_mul(&affine);
        Star {
            representation: new_repr,
            constraints: self.constraints.clone(),
            input_lower_bounds: None,
            input_upper_bounds: None,
        }
    }

    /// Check whether the Star set is empty.
    pub fn is_empty(&self) -> bool {
        if let Some(polytope) = &self.constraints {
            polytope.is_empty()
        } else {
            false
        }
    }

    pub fn get_min(&self, idx: usize) -> T {
        let eqn = self.representation.get_eqn(idx);
        let len = eqn.len();
        let c = &eqn.slice(s![..len - 1]);

        let solved = solve(
            self.constraints
                .as_ref()
                .unwrap()
                .get_coeffs_as_rows()
                .rows(),
            self.constraint_upper_bounds().unwrap(),
            c.view(),
            self.input_lower_bounds.as_ref().map(|x| x.view()),
            self.input_upper_bounds.as_ref().map(|x| x.view()),
        );
        let val = match solved.0 {
            Ok(_) => std::convert::From::from(solved.1.unwrap()),
            Err(ResolutionError::Unbounded) => T::infinity(),
            _ => panic!(),
        };
        self.center()[idx] - val
    }

    pub fn get_max(&self, idx: usize) -> T {
        let neg_one: T = std::convert::From::from(-1.);
        let eqn = self.representation.get_eqn(idx);
        let len = eqn.len();
        let c = &eqn.slice(s![..len - 1]) * neg_one;

        let solved = solve(
            self.constraints
                .as_ref()
                .unwrap()
                .get_coeffs_as_rows()
                .rows(),
            self.constraint_upper_bounds().unwrap(),
            c.view(),
            self.input_lower_bounds.as_ref().map(|x| x.view()),
            self.input_upper_bounds.as_ref().map(|x| x.view()),
        );

        let val = match solved.0 {
            Ok(_) => std::convert::From::from(solved.1.unwrap()),
            Err(ResolutionError::Unbounded) => T::infinity(),
            _ => panic!(),
        };
        self.center()[idx] + val
    }

    /// TODO: doc this
    pub fn trunc_gaussian_cdf(
        &self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
    ) -> (f64, f64, f64) {
        if let Some(poly) = &self.constraints {
            let mu = mu.mapv(|x| x.into());
            let sigma = sigma.mapv(|x| x.into());

            let constraint_coeffs = poly.coeffs().mapv(|x| x.into());
            let upper_bounds = poly.upper_bounds().mapv(|x| x.into());
            let mut sigma_star = constraint_coeffs.t().dot(&sigma.dot(&constraint_coeffs));
            let pos_def_guarator = Array2::from_diag(&Array1::from_elem(sigma_star.nrows(), 1e-12));
            sigma_star = sigma_star + pos_def_guarator;
            let ub = &upper_bounds - &mu.dot(&constraint_coeffs);
            let lb = Array1::from_elem(ub.len(), f64::NEG_INFINITY);
            let out = mv_truncnormal_cdf(lb, ub, sigma_star, n);
            out
        } else {
            (1., 0., 1.)
        }
    }

    pub fn trunc_gaussian_sample(
        &self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
    ) -> Vec<Array1<f64>> {
        if let Some(poly) = &self.constraints {
            let mu = mu.mapv(|x| x.into());
            let sigma = sigma.mapv(|x| x.into());

            let constraint_coeffs = poly.coeffs().mapv(|x| x.into());
            let upper_bounds = poly.upper_bounds().mapv(|x| x.into());

            let mut sigma_star = constraint_coeffs.t().dot(&sigma.dot(&constraint_coeffs));
            let pos_def_guarator = Array2::from_diag(&Array1::from_elem(sigma_star.nrows(), 1e-12));
            sigma_star = sigma_star + pos_def_guarator;
            let ub = &upper_bounds - &mu.dot(&constraint_coeffs);
            let lb = Array1::from_elem(ub.len(), f64::NEG_INFINITY);
            let centered_samples = mv_truncnormal_rand(lb, ub, sigma_star, n);
            let inv_constraint_coeffs = pinv(&constraint_coeffs);

            let samples = centered_samples.dot(&inv_constraint_coeffs) + mu;
            samples
                .rows()
                .into_iter()
                .filter(|x| poly.is_member(&x.mapv(|v| v.into()).view()))
                .map(|x| x.into_owned())
                .collect()
        } else {
            panic!()
        }
    }

    pub fn step_relu(&self, index: usize) -> Vec<Star<T>> {
        let neg_one: T = std::convert::From::from(-1.);
        let mut new_constr = Affine {
            matrix: self
                .representation
                .get_eqn(index)
                .to_owned()
                .insert_axis(Axis(1))
                * neg_one,
            is_lhs: self.representation.is_lhs,
        };
        let neg_shift_part = &new_constr.get_shift() * neg_one;
        new_constr.get_shift_mut().assign(&neg_shift_part);

        let upper_star = self.clone().add_constraints(&new_constr);

        new_constr.matrix = new_constr.matrix * neg_one;
        let mut lower_star = self.clone().add_constraints(&new_constr);
        lower_star
            .representation
            .get_eqn_mut(index)
            .fill(num::zero());
        vec![lower_star, upper_star]
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_map_works() {
        let star = Star::default(2);
        let affine = Affine::new(Array2::<f64>::ones([2, 4]), Array1::<f64>::ones([4]));
        let new_star = star.affine_map(&affine);
        let affine2 = Affine::new(Array2::<f64>::ones([4, 8]), Array1::<f64>::ones([8]));
        let new_new_star = new_star.affine_map(&affine2);
        assert_eq!(new_new_star.input_space_dim(), 2);
        assert_eq!(new_new_star.representation_space_dim(), 8);
    }

    #[test]
    fn trunc_gaussian_cdf_works() {
        let constraint: Affine<f64> = Affine::from_raw(Array2::ones((3, 1)), false);
        let star = Star::new(Array2::ones([2, 4]), Array1::zeros(4)).add_constraints(&constraint);
        star.trunc_gaussian_cdf(&Array1::zeros(2), &(&Array2::eye(2) * 3.), 1000);
    }

    #[test]
    fn step_relu_works() {
        let star: Star<f64> = Star::new(Array2::eye(3), Array1::zeros(3));
        let _child_stars = star.step_relu(0);
    }
}

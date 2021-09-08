#![allow(clippy::module_name_repetitions, clippy::similar_names)]
//! Implementation of [star sets](https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39)
//! for representing affine transformed sets
use crate::affine::{Affine, Affine2, Affine4};
use crate::bounds::Bounds1;
use crate::inequality::Inequality;
use crate::polytope::Polytope;
use crate::tensorshape::TensorShape;
use crate::util::solve;
use good_lp::ResolutionError;
use log::{error, trace};
use ndarray::concatenate;
use ndarray::Array4;
use ndarray::Dimension;
use ndarray::Ix4;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{Axis, Ix2, Zip};
use num::Float;
use rand::Rng;
use std::fmt::Debug;

pub type Star2<A> = Star<A, Ix2>;
pub type Star4<A> = Star<A, Ix4>;

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
/// Based on: Tran, Hoang-Dung, et al. "Star-based reachability analysis of
/// deep neural networks." International Symposium on Formal Methods. Springer,
/// Cham, 2019.
#[derive(Clone, Debug)]
pub struct Star<T: Float, D: Dimension> {
    /// `representation` is the concatenation of [basis center] (where
    /// center is a column vector) and captures information about the
    /// transformed set
    representation: Affine<T, D>,
    /// `constraints` is the concatenation of [coeffs upper_bounds]
    /// and is a representation of the input polyhedron
    constraints: Option<Polytope<T>>,
}

impl<T: Float, D: Dimension> Star<T, D> {
    pub fn ndim(&self) -> usize {
        self.representation.ndim()
    }

    pub fn input_space_polytope(&self) -> Option<&Polytope<T>> {
        self.constraints.as_ref()
    }

    pub fn center(&self) -> ArrayView1<T> {
        self.representation.shift()
    }
}

impl<T: Float, D: Dimension> Star<T, D>
where
    T: ScalarOperand + From<f64> + Debug,
    f64: From<T>,
{
    pub fn num_constraints(&self) -> usize {
        match &self.constraints {
            Some(polytope) => polytope.num_constraints(),
            None => 0,
        }
    }

    /// TODO: doc this
    ///
    /// # Panics
    pub fn trunc_gaussian_cdf(
        &self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
    ) -> (f64, f64, f64) {
        self.constraints
            .as_ref()
            .map_or((1., 0., 1.), |input_region| {
                input_region.gaussian_cdf(mu, sigma, n, max_iters)
            })
    }
}

/*
impl<T: Float> Star<T, IxDyn>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    /// # Panics
    pub fn gaussian_sample<R: Rng>(
        self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        input_bounds: &Option<Bounds1<T>>,
    ) -> Vec<(Array1<T>, T)> {
        match self.ndim() {
            2 => {
                let star: Star2<T> = self.into_dimensionality::<Ix2>().unwrap();
                star.gaussian_sample(rng, mu, sigma, n, max_iters, input_bounds)
                    .into_iter()
                    .map(|(sample, logp)| (sample.mapv(|x| x.into()), logp.into()))
                    .collect()
            }
            _ => panic!(),
        }
    }

    /// # Panics
    pub fn step_relu(self, idx: usize) -> Vec<Self> {
        match self.ndim() {
            2 => {
                let star: Star2<T> = self.into_dimensionality::<Ix2>().unwrap();
                star.step_relu2(idx)
                    .into_iter()
                    .map(Star::into_dyn)
                    .collect()
            }
            _ => panic!(),
        }
    }

    /// # Panics
    pub fn affine_map(self, aff: Affine<T, IxDyn>) -> Self {
        match self.ndim() {
            2 => {
                let star: Star2<T> = self.into_dimensionality::<Ix2>().unwrap();
                let aff: Affine2<T> = aff.into_dimensionality::<Ix2>().unwrap();
                star.affine_map2(&aff).into_dyn()
            }
            _ => panic!(),
        }
    }

    /// # Panics
    pub fn get_min(self, idx: usize) -> T {
        match self.ndim() {
            2 => self.into_dimensionality::<Ix2>().unwrap().get_min(idx),
            _ => panic!(),
        }
    }

    /// # Panics
    pub fn get_max(self, idx: usize) -> T {
        match self.ndim() {
            2 => self.into_dimensionality::<Ix2>().unwrap().get_max(idx),
            _ => panic!(),
        }
    }

    /// # Errors
    pub fn into_dimensionality<D: Dimension>(self) -> Result<Star<T, D>, ShapeError> {
        let constraints = self.constraints;
        self.representation
            .into_dimensionality::<D>()
            .map(|representation| Star {
                representation,
                constraints,
            })
    }
}
*/

impl<T: 'static + Float> Star2<T> {
    /// Create a new Star with given dimension.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    ///
    /// # Panics
    pub fn default(input_shape: &TensorShape) -> Self {
        assert_eq!(input_shape.rank(), 1);
        assert!(input_shape.is_fully_defined());
        let dim = input_shape[0].unwrap();
        Self {
            representation: Affine2::new(Array2::eye(dim), Array1::zeros(dim)),
            constraints: None,
        }
    }

    /// Create a new Star with given basis vector and center.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    pub fn new(basis: Array2<T>, center: Array1<T>) -> Self {
        Self {
            representation: Affine2::new(basis, center),
            constraints: None,
        }
    }

    pub fn with_constraints(mut self, constraints: Polytope<T>) -> Self {
        if self.constraints.is_some() {
            panic!();
        }
        self.constraints = Some(constraints);
        self
    }

    /// Get the dimension of the input space
    pub fn input_space_dim(&self) -> usize {
        self.representation.input_dim()
    }

    /// Get the dimension of the representation space
    pub fn representation_space_dim(&self) -> usize {
        self.representation.output_dim()
    }
}

impl<T: Float> Star2<T>
where
    T: ScalarOperand + From<f64> + Debug,
    f64: From<T>,
{
    pub fn with_input_bounds(mut self, input_bounds: Bounds1<T>) -> Self {
        if self.constraints.is_some() {
            self.constraints = self.constraints.map(|x| x.with_input_bounds(input_bounds));
        } else {
            self.constraints = Some(Polytope::from(input_bounds));
        }
        self
    }
}

impl<T: Float> Star2<T>
where
    T: ndarray::ScalarOperand,
{
    /// Apply an affine transformation to the representation
    pub fn affine_map2(&self, affine: &Affine<T, Ix2>) -> Self {
        Self {
            representation: affine * &self.representation,
            constraints: self.constraints.clone(),
        }
    }
}

impl<T: Float> Star2<T>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    pub fn step_relu2(&self, index: usize) -> Vec<Self> {
        let neg_one: T = std::convert::From::from(-1.);

        let mut new_constr: Inequality<T> = {
            let mut aff = self.representation.get_eqn(index) * neg_one;
            let neg_shift_part = &aff.shift() * neg_one;
            aff.shift_mut().assign(&neg_shift_part);
            aff.into()
        };
        let upper_star = self.clone().add_constraints(&new_constr);

        new_constr *= neg_one;
        let mut lower_star = self.clone().add_constraints(&new_constr);
        lower_star.representation.zero_eqn(index);
        vec![lower_star, upper_star]
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect()
    }

    pub fn step_relu2_dropout(&self, index: usize) -> Vec<Self> {
        let mut dropout_star = self.clone();
        dropout_star.representation.zero_eqn(index);

        let mut stars = self.step_relu2(index);
        stars.insert(0, dropout_star);
        stars.into_iter().filter(|x| !x.is_empty()).collect()
    }

    /// Calculates the minimum value of the equation at index `idx`
    /// given the constraints
    ///
    /// This method assumes that the constraints bound each dimension,
    /// both lower and upper.
    ///
    /// # Panics
    /// TODO: Change output type to Option<T>
    ///
    /// TODO: ResolutionError::Unbounded can result whether or not the
    /// constraints are infeasible if there are zeros in the
    /// objective. This needs to be checked either here or in the
    /// solve function. Currently this is way too hard to do, so we
    /// panic instead. We have an assumption that we start with a
    /// bounded box and therefore should never be unbounded.
    pub fn get_min(&self, idx: usize) -> T {
        let eqn = self.representation.get_eqn(idx);

        if let Some(ref poly) = self.constraints {
            let solved = solve(
                poly.coeffs().rows(),
                poly.ubs(),
                eqn.basis().index_axis(Axis(0), 0),
            );
            let val = match solved.0 {
                Ok(_) => std::convert::From::from(solved.1.unwrap()),
                Err(ResolutionError::Infeasible) => panic!("Error, infeasible"),
                Err(ResolutionError::Unbounded) => panic!("Error, unbounded"),
                _ => panic!(),
            };
            self.center()[idx] + val
        } else {
            T::neg_infinity()
        }
    }

    /// Calculates the maximum value of the equation at index `idx`
    /// given the constraints
    ///
    /// This method assumes that the constraints bound each dimension,
    /// both lower and upper.
    ///
    /// # Panics
    /// TODO: Change output type to Option<T>
    pub fn get_max(&self, idx: usize) -> T {
        let neg_one: T = std::convert::From::from(-1.);
        let eqn = self.representation.get_eqn(idx) * neg_one;

        if let Some(ref poly) = self.constraints {
            let solved = solve(
                poly.coeffs().rows(),
                poly.ubs(),
                eqn.basis().index_axis(Axis(0), 0),
            );
            let val = match solved.0 {
                Ok(_) => std::convert::From::from(solved.1.unwrap()),
                Err(ResolutionError::Infeasible) => panic!("Error, infeasible"),
                Err(ResolutionError::Unbounded) => panic!("Error, unbounded"),
                _ => panic!(),
            };
            self.center()[idx] - val
        } else {
            T::infinity()
        }
    }

    pub fn calculate_axis_aligned_bounding_box(&self) -> Bounds1<T> {
        let lbs = Array1::from_iter((0..self.representation_space_dim()).map(|x| self.get_min(x)));
        let ubs = Array1::from_iter((0..self.representation_space_dim()).map(|x| self.get_max(x)));
        Bounds1::new(lbs, ubs)
    }

    /// Add constraints to restrict the input set. Each row represents a
    /// constraint and the last column represents the upper bounds.
    pub fn add_constraints(mut self, new_constraints: &Inequality<T>) -> Self {
        // assert_eq!(self.representation.is_lhs, new_constraints.is_lhs);
        if let Some(ref mut constrs) = self.constraints {
            constrs.add_constraints(new_constraints);
        } else {
            self.constraints = Some(Polytope::from_halfspaces(new_constraints.clone()));
        }
        self
    }

    /// Check whether the Star set is empty.
    pub fn is_empty(&self) -> bool {
        self.constraints
            .as_ref()
            .map_or(false, crate::polytope::Polytope::is_empty)
    }

    /// # Panics
    // Allow if_let_else because rng is used in both branches, so closures don't work
    #[allow(clippy::too_many_lines, clippy::option_if_let_else)]
    pub fn gaussian_sample<R: Rng>(
        &self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        input_bounds: &Option<Bounds1<T>>,
    ) -> Vec<(Array1<f64>, f64)> {
        // remove fixed dimensions from mu and sigma
        if let Some(poly) = &self.constraints {
            if let Some(bounds) = input_bounds {
                let lbs = bounds.lower();
                let ubs = bounds.upper();
                let unfixed_idxs = Zip::from(lbs).and(ubs).map_collect(|&lb, &ub| lb != ub);
                let sigma_rows: Vec<ArrayView2<T>> = sigma
                    .rows()
                    .into_iter()
                    .zip(&unfixed_idxs)
                    .filter(|(_row, &fix)| fix)
                    .map(|(row, _fix)| row.insert_axis(Axis(0)))
                    .collect();
                let mut reduced_sigma = concatenate(Axis(0), sigma_rows.as_slice()).unwrap();
                let sigma_cols: Vec<ArrayView2<T>> = reduced_sigma
                    .columns()
                    .into_iter()
                    .zip(&unfixed_idxs)
                    .filter(|(_row, &fix)| fix)
                    .map(|(row, _fix)| row.insert_axis(Axis(1)))
                    .collect();
                reduced_sigma = concatenate(Axis(1), sigma_cols.as_slice()).unwrap();
                let reduced_mu: Array1<T> = mu
                    .into_iter()
                    .zip(&unfixed_idxs)
                    .filter(|(_val, &fix)| fix)
                    .map(|(&val, _fix)| val)
                    .collect();
                let (reduced_poly, (_reduced_lbs, _reduced_ubs)) =
                    poly.reduce_fixed_inputs(&lbs, &ubs);
                reduced_poly.gaussian_sample(rng, &reduced_mu, &reduced_sigma, n, max_iters)
            } else {
                poly.gaussian_sample(rng, mu, sigma, n, max_iters)
            }
        } else {
            // Unbounded sample from Gaussian
            todo!()
        }
    }
}

impl<T: 'static + Float> Star4<T> {
    /// Create a new Star with given dimension.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    ///
    /// # Panics
    pub fn default(input_shape: &TensorShape) -> Self {
        assert_eq!(input_shape.rank(), 3);
        assert!(input_shape.is_fully_defined());
        let shape_slice = input_shape.as_defined_slice().unwrap();
        let slice_exact: [usize; 4] = [
            shape_slice[0],
            shape_slice[1],
            shape_slice[2],
            shape_slice[3],
        ];
        Self {
            representation: Affine4::new(Array4::ones(slice_exact), Array1::zeros(shape_slice[3])),
            constraints: None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::*;
    use proptest::prelude::*;
    use proptest::proptest;
    use std::panic;

    proptest! {
        #[test]
        fn test_get_min_feasible(star in non_empty_star(2,3)) {
            prop_assert!(!star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
            prop_assert!(!star.is_empty(), "Non empty star is empty");
            let result = panic::catch_unwind(|| {
                star.get_min(0);
            });
            prop_assert!(result.is_ok(), "Calculating min resulted in panic for feasible star");
        }

        #[test]
        fn test_get_min_infeasible(star in empty_star(2,1)) {
            prop_assert!(star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
            prop_assert!(star.is_empty(), "Empty star is not empty");
            let result = panic::catch_unwind(|| {
                star.get_min(0)
            });
            prop_assert!(result.is_err(), "Infeasible star did not panic for get_min {:#?}", result);
        }

        #[test]
        fn test_get_max_feasible(star in non_empty_star(2,3)) {
            prop_assert!(!star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
            prop_assert!(!star.is_empty(), "Non empty star is empty");
            let result = panic::catch_unwind(|| {
                star.get_max(0);
            });
            prop_assert!(result.is_ok(), "Calculating min resulted in panic for feasible star");
        }

        #[test]
        fn test_get_max_infeasible(star in empty_star(2,1)) {
            prop_assert!(star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
            prop_assert!(star.is_empty(), "Empty star is not empty");
            let result = panic::catch_unwind(|| {
                star.get_max(0)
            });
            prop_assert!(result.is_err(), "Infeasible star did not panic for get_min {:#?}", result);
        }

        #[test]
        fn test_get_min_box_polytope(basis in array2(2, 2)) {
            let num_dims = 2;
            let box_coeffs = Array2::eye(num_dims);
            let mut box_rhs = Array1::ones(num_dims);
            box_rhs *= 20.;

            let mut box_ineq = Inequality::new(box_coeffs.clone(), box_rhs.clone());
            let lower_box_ineq = Inequality::new(-1. * box_coeffs, box_rhs);

            box_ineq.add_eqns(&lower_box_ineq);
            let poly = Polytope::from_halfspaces(box_ineq);

            let center = arr1(&[0.0, 0.0]);
            let star = Star2::new(basis, center).with_constraints(poly);

            let result = panic::catch_unwind(|| {
                star.get_min(0);
            });
            assert!(result.is_ok());
        }

        #[test]
        fn test_get_max_box_polytope(basis in array2(2, 2)) {
            let num_dims = 2;
            let box_coeffs = Array2::eye(num_dims);
            let mut box_rhs = Array1::ones(num_dims);
            box_rhs *= 20.;

            let mut box_ineq = Inequality::new(box_coeffs.clone(), box_rhs.clone());
            let lower_box_ineq = Inequality::new(-1. * box_coeffs, box_rhs);

            box_ineq.add_eqns(&lower_box_ineq);
            let poly = Polytope::from_halfspaces(box_ineq);

            let center = arr1(&[0.0, 0.0]);
            let star = Star2::new(basis, center).with_constraints(poly);

            let result = panic::catch_unwind(|| {
                star.get_max(0);
            });
            assert!(result.is_ok());
        }
    }
}

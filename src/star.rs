#![allow(clippy::module_name_repetitions, clippy::similar_names, non_snake_case)]
//! Implementation of [star sets](https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39)
//! for representing affine transformed sets
use crate::affine::{Affine, Affine2, Affine4};
use crate::bounds::Bounds1;
use crate::gaussian::GaussianDistribution;
use crate::lp::solve;
use crate::lp::LinearSolution;
use crate::polytope::Polytope;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::array;
use ndarray::ArrayView1;
use ndarray::Dimension;
use ndarray::Ix4;
use ndarray::{Array1, Array2};
use ndarray::{Array4, ArrayView2};
use ndarray::{Axis, Ix2};
use num::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Deref;

pub type Star2 = Star<Ix2>;
pub type Star4 = Star<Ix4>;

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
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Star<D: Dimension> {
    /// `representation` is the concatenation of [basis center] (where
    /// center is a column vector) and captures information about the
    /// transformed set
    representation: Affine<D>,
    /// `constraints` is the concatenation of [coeffs upper_bounds]
    /// and is a representation of the input polyhedron
    constraints: Option<Polytope>,
}

impl<D: Dimension> Star<D> {
    pub fn ndim(&self) -> usize {
        self.representation.ndim()
    }

    pub fn input_space_polytope(&self) -> Option<&Polytope> {
        self.constraints.as_ref()
    }

    pub fn center(&self) -> ArrayView1<NNVFloat> {
        self.representation.shift()
    }

    pub fn get_representation(&self) -> &Affine<D> {
        &self.representation
    }

    pub fn get_representation_mut(&mut self) -> &mut Affine<D> {
        &mut self.representation
    }
}

impl<D: Dimension> Star<D> {
    pub fn num_constraints(&self) -> usize {
        match &self.constraints {
            Some(polytope) => polytope.num_constraints(),
            None => 0,
        }
    }

    /// Add constraints to restrict the input set. Each row represents a
    /// constraint and the last column represents the upper bounds.
    #[must_use]
    pub fn add_constraint(mut self, coeffs: ArrayView1<NNVFloat>, rhs: NNVFloat) -> Self {
        if let Some(ref mut constrs) = self.constraints {
            constrs.add_eqn(coeffs, rhs);
        } else {
            self.constraints =
                Polytope::nonempty_new(&coeffs.to_owned().insert_axis(Axis(0)), &array![rhs]);
        }
        self
    }

    #[must_use]
    /// # Panics
    pub fn remove_constraint(mut self, idx: usize) -> Self {
        self.constraints.as_mut().map_or_else(
            || {
                panic!();
            },
            |constrs| constrs.remove_eqn(idx),
        );
        self
    }
}

impl Star2 {
    pub fn get_constraint_coeffs(&self) -> Option<Array2<NNVFloat>> {
        self.constraints.as_ref().map(|x| x.coeffs().to_owned())
    }

    #[must_use]
    pub fn get_safe_subset(&self, safe_value: NNVFloat) -> Self {
        let mut subset = self.clone();
        let mut new_constr: Polytope = self.representation.clone().into();
        let mut rhs = new_constr.rhs_mut();
        rhs *= -1.;
        rhs += safe_value;
        subset.intersect_input(&new_constr);
        subset
    }

    pub fn intersect_input(&mut self, other: &Polytope) {
        self.constraints = self
            .constraints
            .as_ref()
            .map_or(Some(other.clone()), |x| Some(x.intersect(other)));
    }

    /// # Panics
    /// TODO
    pub fn get_input_trunc_gaussian(
        &self,
        mu: ArrayView1<NNVFloat>,
        sigma: ArrayView2<NNVFloat>,
        max_accept_reject_iters: usize,
        stability_eps: NNVFloat,
        bounds_opt: &Option<Bounds1>,
    ) -> Option<GaussianDistribution> {
        todo!();
        // self.constraints
        //     .as_ref()
        //     .and_then(|x| x.reduce_fixed_inputs(bounds_opt))
        //     .map(|poly| {
        //         poly.get_truncnorm_distribution(mu, sigma, max_accept_reject_iters, stability_eps)
        //     })
    }

    /// Create a new Star with given dimension.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    ///
    /// # Panics
    pub fn default(input_shape: &TensorShape) -> Self {
        debug_assert_eq!(input_shape.rank(), 1);
        debug_assert!(input_shape.is_fully_defined());
        let dim = input_shape[0].unwrap();
        Self {
            representation: Affine2::new(Array2::eye(dim), Array1::zeros(dim)),
            constraints: None,
        }
    }

    /// Create a new Star with given basis vector and center.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    pub fn new(basis: Array2<NNVFloat>, center: Array1<NNVFloat>) -> Self {
        Self {
            representation: Affine2::new(basis, center),
            constraints: None,
        }
    }

    /// # Panics
    #[must_use]
    pub fn with_constraints(mut self, constraints: Polytope) -> Self {
        debug_assert!(self.constraints.is_none(), "explicit panic");
        self.constraints = Some(constraints);
        self
    }

    /// Get the dimension of the input space
    pub fn input_dim(&self) -> usize {
        self.representation.input_dim()
    }

    /// Get the dimension of the representation space
    pub fn representation_space_dim(&self) -> usize {
        self.representation.output_dim()
    }
}

impl Star2 {
    /// Apply an affine transformation to the representation
    #[must_use]
    pub fn affine_map2(&self, affine: &Affine<Ix2>) -> Self {
        Self {
            representation: affine * &self.representation,
            constraints: self.constraints.clone(),
        }
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
    /// TODO: `ResolutionError::Unbounded` can result whether or not the
    /// constraints are infeasible if there are zeros in the
    /// objective. This needs to be checked either here or in the
    /// solve function. Currently this is way too hard to do, so we
    /// panic instead. We have an assumption that we start with a
    /// bounded box and therefore should never be unbounded.
    pub fn get_output_min(&self, idx: usize, input_bounds: &Bounds1) -> NNVFloat {
        let eqn = self.representation.get_eqn(idx);
        let shift = eqn.shift()[0];

        self.constraints.as_ref().map_or_else(
            || {
                crate::util::signed_dot(&eqn.basis(), &input_bounds.lower(), &input_bounds.upper())
                    [[0]]
                    + shift
            },
            |poly| {
                let solved = solve(
                    poly.coeffs().rows(),
                    poly.rhs(),
                    eqn.basis().index_axis(Axis(0), 0),
                    Some(input_bounds),
                );
                if let LinearSolution::Solution(_, val) = solved {
                    shift + val
                } else if let LinearSolution::Unbounded(_) = solved {
                    NNVFloat::neg_infinity()
                } else {
                    panic!("Solution: {:?}", solved)
                }
            },
        )
    }

    /// Calculates the maximum value of the equation at index `idx`
    /// given the constraints
    ///
    /// This method assumes that the constraints bound each dimension,
    /// both lower and upper.
    ///
    /// # Panics
    /// TODO: Change output type to Option<T>
    pub fn get_output_max(&self, idx: usize, input_bounds: &Bounds1) -> NNVFloat {
        let eqn = self.representation.get_eqn(idx);
        let shift = eqn.shift()[0];

        self.constraints.as_ref().map_or_else(
            || {
                crate::util::signed_dot(&eqn.basis(), &input_bounds.upper(), &input_bounds.lower())
                    [[0]]
                    + shift
            },
            |poly| {
                let solved = solve(
                    poly.coeffs().rows(),
                    poly.rhs(),
                    eqn.basis().index_axis(Axis(0), 0).mapv(|x| x * -1.).view(),
                    Some(input_bounds),
                );
                if let LinearSolution::Solution(_, val) = solved {
                    shift - val
                } else if let LinearSolution::Unbounded(_) = solved {
                    NNVFloat::infinity()
                } else {
                    panic!()
                }
            },
        )
    }

    //pub fn nearest_euclidean_neighbor(&self, origin: Array1<NNVFloat>) -> Array1<NNVFloat> {}

    /// # Panics
    pub fn can_maximize_output_idx(&self, class_idx: usize) -> bool {
        let class_eqn = self.representation.get_eqn(class_idx);
        let (class_coeffs, class_shift): (ArrayView1<NNVFloat>, NNVFloat) = {
            (
                class_eqn.basis().remove_axis(Axis(0)),
                class_eqn.shift()[[0]],
            )
        };
        let nvars = class_coeffs.len();

        if self.constraints.is_none() {
            return true; // We should have more sophisticated handling here, but this is a stopgap
        }
        let poly = self.constraints.as_ref().unwrap();
        let block_coeffs = poly.coeffs();
        let (A, b) = (block_coeffs.rows(), poly.rhs());
        // Add a constraint for each output class not equal to the one being maximized
        let mut coeffs = Vec::new();
        let mut shifts = Vec::new();
        for idx in 0..self.representation.shift().ndim() {
            if idx == class_idx {
                continue;
            }
            {
                let (diff_coeffs, diff_shift) = {
                    let other_class_eqn = self.representation.get_eqn(idx);
                    (
                        &other_class_eqn.basis().row(0) - &class_coeffs,
                        class_shift - other_class_eqn.shift()[[0]],
                    )
                };
                coeffs.push(diff_coeffs);
                shifts.push(diff_shift);
            }
        }

        let solve_a = A
            .into_iter()
            .chain(coeffs.iter().map(ndarray::ArrayBase::view));
        let solved = solve(
            solve_a,
            b.iter().chain(shifts.iter()),
            Array1::ones(nvars).view(),
            Some(&Bounds1::trivial(nvars)),
        );
        matches!(
            solved,
            LinearSolution::Solution(..) | LinearSolution::Unbounded(..)
        )
    }

    pub fn calculate_output_axis_aligned_bounding_box(
        &self,
        input_outer_bounds: &Bounds1,
    ) -> Bounds1 {
        let lbs = Array1::from_iter(
            (0..self.representation_space_dim())
                .map(|x| self.get_output_min(x, input_outer_bounds)),
        );
        let ubs = Array1::from_iter(
            (0..self.representation_space_dim())
                .map(|x| self.get_output_max(x, input_outer_bounds)),
        );
        Bounds1::new(lbs.view(), ubs.view())
    }

    /// Check whether the Star set is empty.
    pub fn is_empty<Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        input_bounds_opt: Option<Bounds1Ref>,
    ) -> bool {
        self.constraints
            .as_ref()
            .map_or(false, |x| x.is_empty(input_bounds_opt))
    }
}

impl Star4 {
    /// Create a new Star with given dimension.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
    ///
    /// # Panics
    pub fn default(input_shape: &TensorShape) -> Self {
        debug_assert_eq!(input_shape.rank(), 3);
        debug_assert!(input_shape.is_fully_defined());
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
    use ndarray::s;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_add_constraint(star in generic_non_empty_star(4, 4), arr in array1(4)) {
            let ndim = star.input_dim();
            let coeffs = arr.slice(s![0..ndim]);
            let rhs = arr[[0]];
            let num_input_constraints = star.input_space_polytope().map_or(0, |p| p.num_dims());
            let new_star = star.clone().add_constraint(coeffs, rhs);
            let num_output_constraints = new_star.input_space_polytope().map_or(0, |p| p.num_dims());
            prop_assert_eq!(num_input_constraints + 1, num_output_constraints, "Input star: {:?}, Star with constraint: {:?}", star, new_star);
        }
    }
    /*
        #[test]
        fn test_gaussian_sample_manual() {
            let mut rng = Pcg64::seed_from_u64(2);
            let loc: Array1<f64> = arr1(&[-2.52]);
            let scale_diag: Array1<f64> = arr1(&[0.00001]);
            let lbs = loc.clone() - 3.5 * scale_diag.clone();
            let ubs = loc.clone() + 3.5 * scale_diag.clone();
            let input_bounds = Bounds1::new(lbs.view(), ubs.view());
            let mut star: Star2<f64> = Star2::new(Array2::eye(1) * 8.016, Array1::zeros(1));
            star = star.add_constraints(&Inequality::new(arr2(&[[-1.], [1.]]), arr1(&[2.52, 0.10])));
            star.gaussian_sample(
                &mut rng,
                &loc,
                &Array2::from_diag(&scale_diag).to_owned(),
                10,
                20,
                &Some(input_bounds),
            );
            /*
            Star { representation: Affine { basis: [[8.016197283509424]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2, shift: [0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1 }, constraints: Some(Polytope { halfspaces: Inequality { coeffs: [[-1.0],
                [1.0],
                [-1.0],
                [1.0],
                [-1.0]], shape=[5, 1], strides=[1, 1], layn
                let lbs = loc.clone() - 3.5 * scale_diag.clone();
                let ubs = loc.clone() + 3.5 * scale_diag.clone();
                let input_bounds = Bounds1::new(lbs.view(), ubs.view());
                star.gaussian_sample(&mut rng, &loc, &Array2::from_diag(&scale_diag).to_owned(), 10, 20, &Some(input_bounds));
            }
            */

            #[test]
            fn test_get_output_min_feasible(star in non_empty_star(2,3)) {
                prop_assert!(!star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
                prop_assert!(!star.is_empty(), "Non empty star is empty");
                let result = panic::catch_unwind(|| {
                    star.get_output_min(0);
                });
                prop_assert!(result.is_ok(), "Calculating min resulted in panic for feasible star");
            }

            #[test]
            fn test_get_output_min_infeasible(star in empty_star(2,1)) {
                prop_assert!(star.input_space_polytope().unwrap().is_empty(), "Polytope is not empty");
                prop_assert!(star.is_empty(), "Empty star is not empty");
                let result = panic::catch_unwind(|| {
                    star.get_output_min(0)
                });
                prop_assert!(result.is_err(), "Infeasible star did not panic for get_output_min {:#?}", result);
            }

            #[test]
            fn test_get_output_max_feasible(star in non_empty_star(2,3)) {
                prop_assert!(!star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
                prop_assert!(!star.is_empty(), "Non empty star is empty");
                let result = panic::catch_unwind(|| {
                    star.get_output_max(0);
                });
                prop_assert!(result.is_ok(), "Calculating min resulted in panic for feasible star");
            }

            #[test]
            fn test_get_output_max_infeasible(star in empty_star(2,1)) {
                prop_assert!(star.input_space_polytope().unwrap().is_empty(), "Polytope is empty");
                prop_assert!(star.is_empty(), "Empty star is not empty");
                let result = panic::catch_unwind(|| {
                    star.get_output_max(0)
                });
                prop_assert!(result.is_err(), "Infeasible star did not panic for get_output_min {:#?}", result);
            }

            #[test]
            fn test_get_output_min_box_polytope(basis in array2(2, 2)) {
                let num_dims = 2;
                let box_bounds: Bounds1<f64> = Bounds1::new(Array1::from_elem(num_dims, -20.).view(), Array1::from_elem(num_dims, -20.).view());
                let box_ineq = Inequality::new(Array2::zeros([1, num_dims]), Array1::zeros(1), box_bounds);
                let poly = Polytope::from_halfspaces(box_ineq);

                let center = arr1(&[0.0, 0.0]);
                let star = Star2::new(basis, center).with_constraints(poly);

                let result = panic::catch_unwind(|| {
                    star.get_output_min(0);
                });
                assert!(result.is_ok());
            }

            #[test]
            fn test_get_output_max_box_polytope(basis in array2(2, 2)) {
                let num_dims = 2;
                let box_bounds: Bounds1<f64> = Bounds1::new(Array1::from_elem(num_dims, -20.).view(), Array1::from_elem(num_dims, -20.).view());
                let box_ineq = Inequality::new(Array2::zeros([1, num_dims]), Array1::zeros(1), box_bounds);
                let poly = Polytope::from_halfspaces(box_ineq);

                let center = arr1(&[0.0, 0.0]);
                let star = Star2::new(basis, center).with_constraints(poly);

                let result = panic::catch_unwind(|| {
                    star.get_output_max(0);
                });
                assert!(result.is_ok());
            }
        }
    */
}

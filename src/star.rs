#![allow(clippy::module_name_repetitions, clippy::similar_names)]
//! Implementation of [star sets](https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39)
//! for representing affine transformed sets
use crate::affine::{Affine, Affine2, Affine4};
use crate::bounds::Bounds1;
use crate::gaussian::GaussianDistribution;
use crate::inequality::Inequality;
use crate::lp::solve;
use crate::lp::LinearSolution;
use crate::polytope::Polytope;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::Array4;
use ndarray::ArrayView1;
use ndarray::Dimension;
use ndarray::Ix4;
use ndarray::{Array1, Array2};
use ndarray::{Axis, Ix2};
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
pub struct Star<T: NNVFloat, D: Dimension> {
    /// `representation` is the concatenation of [basis center] (where
    /// center is a column vector) and captures information about the
    /// transformed set
    representation: Affine<T, D>,
    /// `constraints` is the concatenation of [coeffs upper_bounds]
    /// and is a representation of the input polyhedron
    constraints: Option<Polytope<T>>,
}

impl<T: NNVFloat, D: Dimension> Star<T, D> {
    pub fn ndim(&self) -> usize {
        self.representation.ndim()
    }

    pub fn input_space_polytope(&self) -> Option<&Polytope<T>> {
        self.constraints.as_ref()
    }

    pub fn center(&self) -> ArrayView1<T> {
        self.representation.shift()
    }

    pub fn get_representation(&self) -> &Affine<T, D> {
        &self.representation
    }
}

impl<T: NNVFloat, D: Dimension> Star<T, D> {
    pub fn num_constraints(&self) -> usize {
        match &self.constraints {
            Some(polytope) => polytope.num_constraints(),
            None => 0,
        }
    }

    /// Add constraints to restrict the input set. Each row represents a
    /// constraint and the last column represents the upper bounds.
    pub fn add_constraints(
        mut self,
        new_constraints: &Inequality<T>,
        check_redundant: bool,
    ) -> Self {
        if let Some(ref mut constrs) = self.constraints {
            constrs.add_constraints(new_constraints, check_redundant);
        } else {
            self.constraints = Some(Polytope::from_halfspaces(new_constraints.clone()));
        }
        self
    }
}

impl<T: NNVFloat> Star2<T> {
    pub fn get_constraint_coeffs(&self) -> Option<Array2<T>> {
        self.constraints.as_ref().map(|x| x.coeffs().to_owned())
    }

    pub fn get_safe_subset(&self, safe_value: T) -> Self {
        let subset = self.clone();
        let mut new_constr: Inequality<T> = self.representation.clone().into();
        let mut rhs = new_constr.rhs_mut();
        rhs *= T::neg(T::one());
        rhs += safe_value;
        subset.add_constraints(&new_constr, false)
    }

    pub fn get_input_trunc_gaussian(
        &self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        max_accept_reject_iters: usize,
        stability_eps: T,
    ) -> Option<GaussianDistribution<T>> {
        self.constraints
            .as_ref()
            .and_then(Polytope::reduce_fixed_inputs)
            .map(|poly| {
                poly.get_truncnorm_distribution(mu, sigma, max_accept_reject_iters, stability_eps)
            })
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
    pub fn new(basis: Array2<T>, center: Array1<T>) -> Self {
        Self {
            representation: Affine2::new(basis, center),
            constraints: None,
        }
    }

    /// # Panics
    pub fn with_constraints(mut self, constraints: Polytope<T>) -> Self {
        debug_assert!(!self.constraints.is_some(), "explicit panic");
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

impl<T: NNVFloat> Star2<T> {
    pub fn get_input_bounds(&self) -> Option<&Bounds1<T>> {
        self.constraints.as_ref().map(Polytope::get_bounds)
    }

    pub fn with_input_bounds(mut self, input_bounds: Bounds1<T>) -> Self {
        if self.constraints.is_some() {
            self.constraints = self.constraints.map(|x| x.with_input_bounds(input_bounds));
        } else {
            self.constraints = Some(Polytope::from(input_bounds));
        }
        self
    }
}

impl<T: NNVFloat> Star2<T> {
    /// Apply an affine transformation to the representation
    pub fn affine_map2(&self, affine: &Affine<T, Ix2>) -> Self {
        Self {
            representation: affine * &self.representation,
            constraints: self.constraints.clone(),
        }
    }
}

impl<T: NNVFloat> Star2<T> {
    pub fn step_relu2(&self, index: usize) -> (Option<Self>, Option<Self>) {
        let neg_one: T = std::convert::From::from(-1.);

        let mut new_constr: Inequality<T> = {
            let mut aff = self.representation.get_eqn(index);
            let neg_basis_part = &aff.basis() * neg_one;
            aff.basis_mut().assign(&neg_basis_part);
            aff.into()
        };
        let upper_star = self.clone().add_constraints(&new_constr, true);

        new_constr *= neg_one;
        let mut lower_star = self.clone().add_constraints(&new_constr, true);
        lower_star.representation.zero_eqn(index);

        let lower_star_opt = if lower_star.is_empty() {
            None
        } else {
            Some(lower_star)
        };
        let upper_star_opt = if upper_star.is_empty() {
            None
        } else {
            Some(upper_star)
        };
        (lower_star_opt, upper_star_opt)
    }

    pub fn step_relu2_dropout(&self, index: usize) -> (Option<Self>, Option<Self>, Option<Self>) {
        let mut dropout_star = self.clone();
        dropout_star.representation.zero_eqn(index);

        let stars = self.step_relu2(index);
        let dropout_star_opt = if dropout_star.is_empty() {
            None
        } else {
            Some(dropout_star)
        };
        (dropout_star_opt, stars.0, stars.1)
    }

    #[cfg(feature = "rayon")]
    pub fn par_get_min(&self, idx: usize) -> T {
        let eqn = self.representation.get_eqn(idx);

        if let Some(ref poly) = self.constraints {
            let solved = par_solve(
                poly.coeffs().rows(),
                poly.ubs(),
                eqn.basis().index_axis(Axis(0), 0),
            );
            if let LinearSolution::Solution(_, val) = solved {
                eqn.shift()[0] + val.into()
            } else {
                panic!()
            }
        } else {
            T::neg_infinity()
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
    pub fn get_min(&self, idx: usize) -> T {
        let eqn = self.representation.get_eqn(idx);

        self.constraints.as_ref().map_or_else(
            || T::neg_infinity(),
            |poly| {
                let solved = solve(
                    poly.coeffs().rows(),
                    poly.ubs(),
                    eqn.basis().index_axis(Axis(0), 0),
                    poly.get_bounds(),
                );
                if let LinearSolution::Solution(_, val) = solved {
                    eqn.shift()[0] + val.into()
                } else {
                    panic!("Solution: {:?}", solved)
                }
            },
        )
    }

    #[cfg(feature = "rayon")]
    pub fn par_get_max(&self, idx: usize) -> T {
        let neg_one: T = std::convert::From::from(-1.);
        let mut eqn = self.representation.get_eqn(idx);
        let shift = eqn.shift()[0];
        eqn *= neg_one;

        if let Some(ref poly) = self.constraints {
            let solved = par_solve(
                poly.coeffs().rows(),
                poly.ubs(),
                eqn.basis().index_axis(Axis(0), 0),
            );
            if let LinearSolution::Solution(_, val) = solved {
                shift - val.into()
            } else {
                panic!()
            }
        } else {
            T::infinity()
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
        let mut eqn = self.representation.get_eqn(idx);
        let shift = eqn.shift()[0];
        eqn *= neg_one;

        self.constraints.as_ref().map_or_else(
            || T::infinity(),
            |poly| {
                let solved = solve(
                    poly.coeffs().rows(),
                    poly.ubs(),
                    eqn.basis().index_axis(Axis(0), 0),
                    poly.get_bounds(),
                );
                if let LinearSolution::Solution(_, val) = solved {
                    shift - val.into()
                } else {
                    panic!()
                }
            },
        )
    }

    pub fn calculate_axis_aligned_bounding_box(&self) -> Bounds1<T> {
        let lbs = Array1::from_iter((0..self.representation_space_dim()).map(|x| self.get_min(x)));
        let ubs = Array1::from_iter((0..self.representation_space_dim()).map(|x| self.get_max(x)));
        Bounds1::new(lbs.view(), ubs.view())
    }

    /// Check whether the Star set is empty.
    pub fn is_empty(&self) -> bool {
        self.constraints
            .as_ref()
            .map_or(false, crate::polytope::Polytope::is_empty)
    }
}

impl<T: NNVFloat> Star4<T> {
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
    use crate::test_util::{array2, empty_star, non_empty_star};
    use ndarray::arr1;
    use proptest::prelude::*;
    use proptest::proptest;
    use std::panic;

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
                prop_assert!(star.input_space_polytope().unwrap().is_empty(), "Polytope is not empty");
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
                let box_bounds: Bounds1<f64> = Bounds1::new(Array1::from_elem(num_dims, -20.).view(), Array1::from_elem(num_dims, -20.).view());
                let box_ineq = Inequality::new(Array2::zeros([1, num_dims]), Array1::zeros(1), box_bounds);
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
                let box_bounds: Bounds1<f64> = Bounds1::new(Array1::from_elem(num_dims, -20.).view(), Array1::from_elem(num_dims, -20.).view());
                let box_ineq = Inequality::new(Array2::zeros([1, num_dims]), Array1::zeros(1), box_bounds);
                let poly = Polytope::from_halfspaces(box_ineq);

                let center = arr1(&[0.0, 0.0]);
                let star = Star2::new(basis, center).with_constraints(poly);

                let result = panic::catch_unwind(|| {
                    star.get_max(0);
                });
                assert!(result.is_ok());
            }
        }
    */
}

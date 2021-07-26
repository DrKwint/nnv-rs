//! Implementation of [star sets](https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39)
//! for representing affine transformed sets
use crate::affine::{Affine, Affine2, Affine4};
use crate::inequality::Inequality;
use crate::polytope::Polytope;
use crate::tensorshape::TensorShape;
use crate::util::solve;
use good_lp::ResolutionError;
use ndarray::concatenate;
use ndarray::Array4;
use ndarray::Dimension;
use ndarray::Ix4;
use ndarray::ShapeError;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{Axis, Ix2, IxDyn, Zip};
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
///
/// Shapes
/// basis - \[input_dim, repr_dim\]
/// center - \[repr_dim\]
/// constraints - \[num_constraints, input_dim\]
/// upper_bounds - \[num_constraints\]
#[derive(Clone, Debug)]
pub struct Star<T: Float, D: Dimension> {
    /// `representation` is the concatenation of [basis center] (where center is a column vector) and captures information about the transformed set
    representation: Affine<T, D>,
    /// `constraints` is the concatenation of [coeffs upper_bounds] and is a representation of the input polyhedron
    constraints: Option<Polytope<T>>,
}

impl<T: Float, D: Dimension> Star<T, D>
where
    T: std::convert::From<f64> + std::fmt::Debug + std::fmt::Display + ndarray::ScalarOperand,
    f64: std::convert::From<T>,
{
    pub fn ndim(&self) -> usize {
        self.representation.ndim()
    }

    pub fn input_space_polytope(&self) -> Option<&Polytope<T>> {
        self.constraints.as_ref()
    }

    pub fn center(&self) -> ArrayView1<T> {
        self.representation.shift()
    }

    pub fn num_constraints(&self) -> usize {
        match &self.constraints {
            Some(polytope) => polytope.num_constraints(),
            None => 0,
        }
    }

    pub fn into_dyn(self) -> Star<T, IxDyn> {
        Star {
            representation: self.representation.into_dyn(),
            constraints: self.constraints,
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
        if let Some(input_region) = &self.constraints {
            input_region.gaussian_cdf(mu, sigma, n, max_iters)
        } else {
            // No constraints means the entire distribution is in the input region
            (1., 0., 1.)
        }
    }
}

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
    pub fn gaussian_sample<R: Rng>(
        self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        input_bounds: &Option<(Array1<T>, Array1<T>)>,
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

    pub fn step_relu(self, idx: usize) -> Vec<Self> {
        match self.ndim() {
            2 => {
                let star: Star2<T> = self.into_dimensionality::<Ix2>().unwrap();
                star.step_relu2(idx)
                    .into_iter()
                    .map(|x| x.into_dyn())
                    .collect()
            }
            _ => panic!(),
        }
    }

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

    pub fn get_min(self, idx: usize) -> T {
        match self.ndim() {
            2 => self.into_dimensionality::<Ix2>().unwrap().get_min(idx),
            _ => panic!(),
        }
    }

    pub fn get_max(self, idx: usize) -> T {
        match self.ndim() {
            2 => self.into_dimensionality::<Ix2>().unwrap().get_max(idx),
            _ => panic!(),
        }
    }

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
    /// Create a new Star with given dimension.
    ///
    /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
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

    pub fn with_input_bounds(mut self, lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        if self.constraints.is_some() {
            self.constraints = self
                .constraints
                .map(|x| x.with_input_bounds(lower_bounds, upper_bounds));
        } else {
            self.constraints = Some(Polytope::from_input_bounds(lower_bounds, upper_bounds));
        }
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

    /// Apply an affine transformation to the representation
    pub fn affine_map2(&self, affine: &Affine<T, Ix2>) -> Self {
        Self {
            representation: affine * &self.representation,
            constraints: self.constraints.clone(),
        }
    }

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

    /// # Panics
    pub fn get_min(&self, idx: usize) -> T {
        let eqn = self.representation.get_eqn(idx).get_raw_augmented();
        let c = &eqn.index_axis(Axis(0), 0);

        if let Some(ref poly) = self.constraints {
            let solved = solve(poly.coeffs().rows(), poly.ubs(), c.view());
            let val = match solved.0 {
                Ok(_) => std::convert::From::from(solved.1.unwrap()),
                Err(ResolutionError::Unbounded) => T::neg_infinity(),
                _ => panic!(),
            };
            self.center()[idx] + val
        } else {
            T::neg_infinity()
        }
    }

    /// # Panics
    pub fn get_max(&self, idx: usize) -> T {
        let neg_one: T = std::convert::From::from(-1.);
        let eqn = self.representation.get_eqn(idx).get_raw_augmented();
        let c = &eqn.index_axis(Axis(0), 0) * neg_one;

        if let Some(ref poly) = self.constraints {
            let solved = solve(poly.coeffs().rows(), poly.ubs(), c.view());
            let val = match solved.0 {
                Ok(_) => std::convert::From::from(solved.1.unwrap()),
                Err(ResolutionError::Unbounded) => T::neg_infinity(),
                _ => panic!(),
            };
            self.center()[idx] - val
        } else {
            T::infinity()
        }
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

    #[allow(clippy::too_many_lines)]
    pub fn gaussian_sample<R: Rng>(
        &self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        input_bounds: &Option<(Array1<T>, Array1<T>)>,
    ) -> Vec<(Array1<f64>, f64)> {
        // remove fixed dimensions from mu and sigma
        if let Some(poly) = &self.constraints {
            if let Some((lbs, ubs)) = input_bounds {
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
                    poly.reduce_fixed_inputs(lbs, ubs);
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

/*
/// `basis` is the matrix used to transform from the input space to the representation space.
pub fn basis(&self) -> ArrayView2<T> {
    self.representation.get_mul()
}

/// `center` is the column vector which shifts all points in the input and representation polyhedra from the origin.
pub fn center(&self) -> ArrayView1<T> {
    self.representation.get_shift()
}

pub fn constraint_coeffs(&self) -> Option<ArrayView2<T>> {
    self.constraints
        .as_ref()
        .and_then(crate::polytope::Polytope::coeffs)
}

pub fn constraint_upper_bounds(&self) -> Option<ArrayView1<T>> {
    self.constraints
        .as_ref()
        .map(|constrs| constrs.eqn_upper_bounds())
}

/// # Panics
#[allow(clippy::too_many_lines)]
pub fn trunc_gaussian_sample(
    &self,
    mu: &Array1<T>,
    sigma: &Array2<T>,
    n: usize,
    max_iters: usize,
) -> Vec<(Array1<f64>, f64)> {
    let mut rng = rand::thread_rng();
    if let Some(poly) = &self.constraints {
        // convert T to f64 in inputs
        let mu = mu.mapv(std::convert::Into::into);
        let sigma = sigma.mapv(std::convert::Into::into);

        // remove fixed dimensions from mu and sigma
        let (lbs, ubs) = poly.get_input_bounds().unwrap();
        let unfixed = Zip::from(lbs).and(ubs).map_collect(|&lb, &ub| lb != ub);
        let sigma_rows: Vec<ArrayView2<f64>> = sigma
            .rows()
            .into_iter()
            .zip(&unfixed)
            .filter(|(_row, &fix)| fix)
            .map(|(row, _fix)| row.insert_axis(Axis(0)))
            .collect();
        let mut reduced_sigma = concatenate(Axis(0), sigma_rows.as_slice()).unwrap();
        let sigma_rows: Vec<ArrayView2<f64>> = reduced_sigma
            .columns()
            .into_iter()
            .zip(&unfixed)
            .filter(|(_row, &fix)| fix)
            .map(|(row, _fix)| row.insert_axis(Axis(1)))
            .collect();
        reduced_sigma = concatenate(Axis(1), sigma_rows.as_slice()).unwrap();
        let reduced_mu: Array1<f64> = Array1::from_iter(
            mu.into_iter()
                .zip(unfixed)
                .filter(|(_val, fix)| *fix)
                .map(|(row, _fix)| row),
        );
        let reduced_space_poly = poly.reduce_fixed_inputs();

        // sample unfixed dimensions
        let constraint_coeffs_opt = reduced_space_poly
            .coeffs()
            .map(|x| x.mapv(std::convert::Into::into));
        if let Some(mut constraint_coeffs) = constraint_coeffs_opt {
            // normalise each equation
            let constraint_coeff_norms: Array1<f64> = constraint_coeffs
                .columns()
                .into_iter()
                .map(|col| col.mapv(|x| x * x).sum().sqrt())
                .collect();
            constraint_coeffs /= &constraint_coeff_norms;
            let ub = reduced_space_poly
                .eqn_upper_bounds()
                .mapv(std::convert::Into::into)
                / constraint_coeff_norms;

            // embed constraint coeffs in an identity matrix
            let sq_coeffs = embed_identity(&constraint_coeffs, None).reversed_axes();
            // if there are more constraints than variables, add dummy variables
            let sq_reduced_sigma = embed_identity(&reduced_sigma, Some(sq_coeffs.nrows()));
            let sq_constr_sigma = {
                let sigma: Array2<f64> = sq_coeffs.dot(&sq_reduced_sigma.dot(&sq_coeffs.t()));
                let diag_addn = Array2::from_diag(&Array1::from_elem(sigma.nrows(), 1e-12));
                sigma + diag_addn
            };
            let mut sq_ub = Array::from_elem(sq_coeffs.nrows(), f64::INFINITY);
            sq_ub.slice_mut(s![..ub.len()]).assign(&ub);

            let extended_reduced_mu = if sq_coeffs.nrows() == reduced_mu.len() {
                reduced_mu.clone()
            } else {
                let mut e_r_mu = Array1::zeros(sq_coeffs.nrows());
                e_r_mu.slice_mut(s![..reduced_mu.len()]).assign(&reduced_mu);
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
                .sample(&mut rng);
                (sample.insert_axis(Axis(1)), array![1.])
            } else {
                mv_truncnormal_rand(sq_constr_lb, sq_constr_ub, sq_constr_sigma, n, max_iters)
            };
            let inv_constraint_coeffs = pinv(&sq_coeffs); // TODO: switch to proper inverse
            let mut samples = inv_constraint_coeffs
                .dot(&centered_samples.t())
                .reversed_axes();
            samples = samples
                .slice_axis(Axis(1), Slice::from(0..reduced_sigma.nrows()))
                .to_owned();
            let mut filtered_samples: Vec<(Array1<f64>, f64)> = samples
                .rows()
                .into_iter()
                .zip(logp)
                .map(|(x, logp)| (x.to_owned() + &reduced_mu, logp))
                .filter(|(x, _logp)| reduced_space_poly.in_bounds(&x.mapv(|v| v.into()).view()))
                .collect();
            if filtered_samples.is_empty() {
                let (x_c, _r) = self.constraints.as_ref().unwrap().chebyshev_center();
                filtered_samples = vec![(x_c, 0.43)]
            }
            filtered_samples
        } else {
            let mvn = MultivariateNormal::new(reduced_mu, reduced_sigma);
            let sample = mvn.sample(&mut rng);
            let logp = mvn.logp(&sample);
            vec![(sample, logp)]
        }
    } else {
        panic!()
    }
}

*/

/*
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
        star.trunc_gaussian_cdf(&Array1::zeros(2), &(&Array2::eye(2) * 3.), 1000, 10);
    }

    #[test]
    fn step_relu_works() {
        let star: Star<f64> = Star::new(Array2::eye(3), Array1::zeros(3));
        let _child_stars = star.step_relu(0);
    }
}
*/

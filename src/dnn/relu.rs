use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::graph::Operation;
use crate::star::Star2;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::Zip;
use num::Float;
use num::Zero;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter, Result};
use std::ops::Deref;
use std::ops::Neg;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ReLU {
    ndims: usize,
}

impl ReLU {
    pub const fn new(ndims: usize) -> Self {
        Self { ndims }
    }
}

impl Display for ReLU {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "ReLU")
    }
}

impl Operation for ReLU {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn num_steps(&self) -> Option<usize> {
        Some(self.ndims)
    }

    fn inputs_dims(&self) -> Vec<usize> {
        vec![self.ndims]
    }

    fn outputs_dims(&self) -> Vec<usize> {
        vec![self.ndims]
    }

    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        vec![input[0].mapv(|x| if x.lt(&0.) { 0. } else { x })]
    }

    fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
        vec![input[0].mapv(|x| if x.lt(&0.) { 0. } else { x })]
    }

    fn apply_bounds(
        &self,
        bounds: &[&Bounds1],
        lower_aff: &[&Affine2],
        upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        // if (self.ndims + 1) == bounds[0].ndim() {
        vec![deep_poly_relu(&bounds[0], &lower_aff[0], &upper_aff[0])]
        // } else {
        //     let (bounds_head, bounds_tail) = bounds[0].split_at(self.ndims);
        //     let (lower_aff_head, lower_aff_tail) = lower_aff[0].split_at(self.ndims);
        //     let (upper_aff_head, upper_aff_tail) = lower_aff[0].split_at(self.ndims);
        //     let (bounds_part, lower_part, upper_part) =
        //         deep_poly_relu(&bounds_head, &lower_aff_head, &upper_aff_head);
        //     vec![(
        //         bounds_part.append(&bounds_tail),
        //         lower_part.append(&lower_aff_tail),
        //         upper_part.append(&upper_aff_tail),
        //     )]
        // }
    }

    fn apply_bounds_step(
        &self,
        dim: usize,
        bounds: &[&Bounds1],
        lower_aff: &[&Affine2],
        upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        vec![deep_poly_steprelu(
            dim,
            bounds[0].clone(),
            lower_aff[0].clone(),
            upper_aff[0].clone(),
        )]
    }

    fn get_activation_pattern(&self, state: &[&Array2<NNVFloat>]) -> Option<Vec<Array2<bool>>> {
        // This should only be Some in an activation layer (e.g. ReLU)
        Some(vec![state[0].mapv(|x| x >= 0.0)])
    }

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        stars: Vec<StarRef>,
        dim: Option<usize>,
        input_bounds: &Bounds1,
        parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>> {
        assert!(parent_local_output_bounds_opt
            .as_ref()
            .map_or(true, |b| b.len() == 1));
        let parent_local_output_bounds_opt = parent_local_output_bounds_opt.map(|x| x[0].clone());
        assert_eq!(1, stars.len());
        let star = stars.get(0).unwrap();

        let dim = dim.unwrap();
        let child_stars = star.step_relu2(dim, Some(input_bounds));
        let is_single_child = child_stars.0.is_some() ^ child_stars.1.is_some();

        let mut stars = vec![];
        let mut star_local_output_bounds = Vec::new();
        if let Some(mut lower_star) = child_stars.0 {
            // local_output_bounds output
            star_local_output_bounds.push(parent_local_output_bounds_opt.as_ref().map(
                |parent_local_output_bounds| {
                    let mut local_output_bounds: Bounds1 =
                        Bounds1::clone(&parent_local_output_bounds);
                    local_output_bounds.index_mut(dim)[0] = 0.;
                    local_output_bounds.index_mut(dim)[1] = 0.;
                    local_output_bounds
                },
            ));

            // star output
            if is_single_child {
                // Remove redundant constraint added by step_relu2 above
                let num_constraints = lower_star.num_constraints();
                lower_star = lower_star.remove_constraint(num_constraints - 1);
            }
            stars.push(lower_star);
        }

        if let Some(mut upper_star) = child_stars.1 {
            // local_output_bounds output
            star_local_output_bounds.push(parent_local_output_bounds_opt.map(
                |parent_local_output_bounds| {
                    let mut local_output_bounds: Bounds1 =
                        Bounds1::clone(&parent_local_output_bounds);
                    let mut lb = local_output_bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = 0.;
                    }
                    local_output_bounds
                },
            ));

            // star output
            if is_single_child {
                // Remove redundant constraint added by step_relu2 above
                let num_constraints = upper_star.num_constraints();
                upper_star = upper_star.remove_constraint(num_constraints - 1);
            }
            stars.push(upper_star);
        }

        vec![stars
            .into_iter()
            .zip(star_local_output_bounds.into_iter())
            .collect()]
    }
}

/// # Panics
pub fn deep_poly_steprelu(
    dim: usize,
    mut bounds: Bounds1,
    mut lower_aff: Affine2,
    mut upper_aff: Affine2,
) -> (Bounds1, Affine2, Affine2) {
    let mut bounds_slice = bounds.index_mut(dim);
    let (mut lbasis, mut lshift) = lower_aff.get_eqn_mut(dim);
    let (mut u_basis, mut u_shift) = upper_aff.get_eqn_mut(dim);
    let l = bounds_slice[[0]];
    let u = bounds_slice[[1]];
    if u <= NNVFloat::zero() {
        bounds_slice.fill(NNVFloat::zero());
        lbasis.fill(NNVFloat::zero());
        u_basis.fill(NNVFloat::zero());
        lshift.fill(NNVFloat::zero());
        u_shift.fill(NNVFloat::zero());
    // gt branch
    } else if l >= NNVFloat::zero() {
        // here, leave mul and shift at defaults
        // then, spanning branch
    } else {
        // Using y = ax + b:
        // Handling so things don't break down in the infinite case
        if u == NNVFloat::infinity() {
            u_basis.mapv_inplace(|x| {
                if x * NNVFloat::infinity() == NNVFloat::nan() {
                    0.
                } else {
                    NNVFloat::INFINITY
                }
            });
            u_shift.mapv_inplace(|x| {
                if x * NNVFloat::infinity() == NNVFloat::nan() {
                    0.
                } else {
                    NNVFloat::INFINITY
                }
            });
        } else {
            u_basis.mapv_inplace(|a| a * (u / (u - l)));
            u_shift.mapv_inplace(|b| u * (b - l) / (u - l));
        }

        // use approximation with least area
        if u < NNVFloat::neg(l) || l == NNVFloat::neg_infinity() {
            // Eqn. 3 from the paper
            *bounds_slice.get_mut(0).unwrap() = NNVFloat::zero();
            lbasis.fill(NNVFloat::zero());
            lshift.fill(NNVFloat::zero());
        } else {
            // Eqn. 4 from the paper, leave l_mul at default
        }
    }
    //debug_assert!(bounds.is_all_finite());
    (bounds, lower_aff, upper_aff)
}

pub fn deep_poly_relu(
    bounds: &Bounds1,
    lower_aff: &Affine2,
    upper_aff: &Affine2,
) -> (Bounds1, Affine2, Affine2) {
    let mut out = bounds.clone();
    let mut l_mul = Array1::ones(bounds.ndim());
    let mut u_mul = Array1::ones(bounds.ndim());
    let mut u_shift = Array1::zeros(bounds.ndim());
    Zip::from(bounds.bounds_iter())
        .and(out.bounds_iter_mut())
        .and(&mut l_mul)
        .and(&mut u_mul)
        .and(&mut u_shift)
        .for_each(
            |b: ArrayView1<NNVFloat>,
             mut out: ArrayViewMut1<NNVFloat>,
             l_mul: &mut NNVFloat,
             u_mul: &mut NNVFloat,
             u_shift: &mut NNVFloat| {
                let l = b[0];
                let u = b[1];
                // lt branch
                if u <= NNVFloat::zero() {
                    out[0] = NNVFloat::zero();
                    out[1] = NNVFloat::zero();
                    *l_mul = NNVFloat::zero();
                    *u_mul = NNVFloat::zero();
                // gt branch
                } else if l >= NNVFloat::zero() {
                    // Leave mul and shift at defaults
                    // spanning branch
                } else {
                    *u_mul = u / (u - l);
                    *u_shift = NNVFloat::neg((u * l) / (u - l));
                    // use approximation with least area
                    if u < NNVFloat::neg(l) {
                        // Eqn. 3 from the paper
                        out[0] = NNVFloat::zero();
                        *l_mul = NNVFloat::zero();
                    } else {
                        // Eqn. 4 from the paper, leave l_mul at default
                    }
                }
            },
        );
    let mut lower_aff = lower_aff.clone();
    lower_aff.scale_eqns(l_mul.view());
    let mut upper_aff = upper_aff.clone();
    upper_aff.scale_eqns(u_mul.view());
    upper_aff = upper_aff + u_shift;
    (out, lower_aff, upper_aff)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        // fn test_single_dim_relu(star_basis in array2(1, 1), star_center in array1(1), constraints in polytope_including_zero(1, 10)) {
        fn test_single_dim_relu(input_star in non_empty_star(1, 4)) {
            // let input_star = Star2::new(star_basis, star_center).with_constraints(constraints);
            let bounds = bounds1_set(1, 30.);
            let relu = ReLU::new(1);
            relu.forward_star(vec![&input_star], Some(0), &bounds, Some(vec![&bounds]));
        }
    }

    #[test]
    fn test_deeppoly_relu_gt_correctness() {
        let bounds: Bounds1 = Bounds1::new(Array1::zeros(4).view(), Array1::ones(4).view());
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let (new_b, new_l, new_u) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, bounds);
        assert_eq!(new_l, lower_aff);
        assert_eq!(new_u, upper_aff);
    }

    #[test]
    fn test_deeppoly_relu_lt_correctness() {
        let bounds: Bounds1 = Bounds1::new((Array1::ones(4) * -1.).view(), Array1::zeros(4).view());
        let lower_aff = Affine2::identity(4) + (-4.);
        let upper_aff = Affine2::identity(4);
        let (new_b, new_l, new_u) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(
            new_b,
            Bounds1::new(Array1::zeros(4).view(), Array1::zeros(4).view())
        );
        assert_eq!(new_l, Affine2::identity(4) * 0.);
        assert_eq!(new_u, Affine2::identity(4) * 0.);
    }

    #[test]
    fn test_deeppoly_relu_spanning_firstbranch_correctness() {
        let bounds: Bounds1 = Bounds1::new((Array1::ones(4) * -2.).view(), Array1::ones(4).view());
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let upper_aff_update = Affine2::new(
            Array2::from_diag(&(&bounds.upper() / (&bounds.upper() - &bounds.lower()))),
            &bounds.upper() * &bounds.lower() / (&bounds.upper() - &bounds.lower()) * -1.,
        );
        let (new_b, new_l, new_u) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(
            new_b,
            Bounds1::new(Array1::zeros(4).view(), Array1::ones(4).view())
        );
        assert_eq!(new_l, lower_aff * 0.);
        assert_eq!(new_u, upper_aff * &upper_aff_update);
    }

    #[test]
    fn test_deeppoly_relu_spanning_secondbranch_correctness() {
        let bounds: Bounds1 = Bounds1::new((Array1::ones(4) * -1.).view(), Array1::ones(4).view());
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let upper_aff_update = Affine2::new(
            Array2::from_diag(&(&bounds.upper() / (&bounds.upper() - &bounds.lower()))),
            &bounds.upper() * &bounds.lower() / (&bounds.upper() - &bounds.lower()) * -1.,
        );
        let (new_b, new_l, new_u) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, bounds);
        assert_eq!(new_l, lower_aff);
        assert_eq!(new_u, upper_aff * &upper_aff_update);
    }
}

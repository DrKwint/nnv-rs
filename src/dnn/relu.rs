use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::graph::Operation;
use crate::star::Star2;
use crate::star_node::StarNodeType;
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

#[typetag::serde]
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

    fn forward1(&self, input: &Vec<Array1<NNVFloat>>) -> Vec<Array1<NNVFloat>> {
        vec![input[0].mapv(|x| if x.lt(&0.) { 0. } else { x })]
    }

    fn forward2(&self, input: &Vec<Array2<NNVFloat>>) -> Vec<Array2<NNVFloat>> {
        vec![input[0].mapv(|x| if x.lt(&0.) { 0. } else { x })]
    }

    fn apply_bounds(
        &self,
        bounds: &Vec<Bounds1>,
        lower_aff: &Vec<Affine2>,
        upper_aff: &Vec<Affine2>,
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        if (self.ndims + 1) == bounds[0].ndim() {
            vec![deep_poly_relu(&bounds[0], &lower_aff[0], &upper_aff[0])]
        } else {
            let (bounds_head, bounds_tail) = bounds[0].split_at(self.ndims);
            let (lower_aff_head, lower_aff_tail) = lower_aff[0].split_at(self.ndims);
            let (upper_aff_head, upper_aff_tail) = lower_aff[0].split_at(self.ndims);
            let (bounds_part, lower_part, upper_part) =
                deep_poly_relu(&bounds_head, &lower_aff_head, &upper_aff_head);
            vec![(
                bounds_part.append(&bounds_tail),
                lower_part.append(&lower_aff_tail),
                upper_part.append(&upper_aff_tail),
            )]
        }
    }

    fn apply_bounds_step(
        &self,
        dim: usize,
        bounds: &Vec<Bounds1>,
        lower_aff: &Vec<Affine2>,
        upper_aff: &Vec<Affine2>,
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        vec![deep_poly_steprelu(
            dim,
            bounds[0].clone(),
            lower_aff[0].clone(),
            upper_aff[0].clone(),
        )]
    }

    fn get_activation_pattern(&self, state: Vec<&Array2<NNVFloat>>) -> Option<Vec<Array2<bool>>> {
        // This should only be Some in an activation layer (e.g. ReLU)
        Some(vec![state[0].mapv(|x| x >= 0.0)])
    }

    fn forward_star(
        &self,
        star: &Star2,
        dim: Option<usize>,
        input_bounds: Option<Vec<Bounds1>>,
        parent_bounds: Option<Vec<Bounds1>>,
    ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool) {
        let dim = dim.unwrap();
        let child_stars = star.step_relu2(dim, &input_bounds.map(|x| x[0]));
        let parent_bounds = parent_bounds.unwrap()[0];
        let mut same_output_bounds = false;
        let mut stars = vec![];
        let mut star_input_bounds = vec![];
        let is_single_child = child_stars.0.is_some() ^ child_stars.1.is_some();

        if let Some(mut lower_star) = child_stars.0 {
            let mut bounds = parent_bounds.clone();
            bounds.index_mut(dim)[0] = 0.;
            bounds.index_mut(dim)[1] = 0.;
            if is_single_child {
                // Remove redundant constraint added by step_relu2 above
                let num_constraints = lower_star.num_constraints();
                lower_star = lower_star.remove_constraint(num_constraints - 1);
            }

            stars.push(lower_star);
            star_input_bounds.push(Some(bounds));
        }

        if let Some(mut upper_star) = child_stars.1 {
            let mut bounds = parent_bounds;
            let mut lb = bounds.index_mut(dim);
            if lb[0].is_sign_negative() {
                lb[0] = 0.;
            }
            if is_single_child {
                // Remove redundant constraint added by step_relu2 above
                let num_constraints = upper_star.num_constraints();
                upper_star = upper_star.remove_constraint(num_constraints - 1);
                same_output_bounds = true;
            }
            stars.push(upper_star);
            star_input_bounds.push(Some(bounds));
        }
        (stars, star_input_bounds, same_output_bounds)
    }

    fn construct_starnodetype(&self, child_ids: &[usize], dim: Option<usize>) -> StarNodeType {
        debug_assert_gt!(child_ids.len(), 0);
        StarNodeType::StepRelu {
            dim: dim.unwrap(),
            fst_child_idx: child_ids[0],
            snd_child_idx: child_ids.get(1).copied(),
        }
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

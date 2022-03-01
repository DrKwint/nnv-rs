use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly_relu;
use crate::deeppoly::deep_poly_steprelu;
use crate::dnn::layer::Layer;
use crate::star::Star2;
use crate::star_node::StarNodeType;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ReLU {
    ndims: usize,
}

impl ReLU {
    pub fn new(ndims: usize) -> Self {
        ReLU { ndims }
    }
}

impl Display for ReLU {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "ReLU")
    }
}

#[typetag::serde]
impl Layer for ReLU {
    fn is_activation(&self) -> bool {
        true
    }

    fn input_dims(&self) -> usize {
        self.ndims
    }

    fn output_dims(&self) -> usize {
        self.ndims
    }

    fn forward1(&self, input: &Array1<NNVFloat>) -> Array1<NNVFloat> {
        input.mapv(|x| if x.lt(&0.) { 0. } else { x })
    }

    fn forward2(&self, input: &Array2<NNVFloat>) -> Array2<NNVFloat> {
        input.mapv(|x| if x.lt(&0.) { 0. } else { x })
    }

    fn apply_bounds(
        &self,
        bounds: &Bounds1,
        lower_aff: &Affine2,
        upper_aff: &Affine2,
    ) -> (Bounds1, (Affine2, Affine2)) {
        if (self.ndims + 1) == bounds.ndim() {
            deep_poly_relu(bounds, lower_aff, upper_aff)
        } else {
            let (bounds_head, bounds_tail) = bounds.split_at(self.ndims);
            let (lower_aff_head, lower_aff_tail) = lower_aff.split_at(self.ndims);
            let (upper_aff_head, upper_aff_tail) = lower_aff.split_at(self.ndims);
            let (bounds_part, (lower_part, upper_part)) =
                deep_poly_relu(&bounds_head, &lower_aff_head, &upper_aff_head);
            (
                bounds_part.append(&bounds_tail),
                (
                    lower_part.append(&lower_aff_tail),
                    upper_part.append(&upper_aff_tail),
                ),
            )
        }
    }

    fn apply_bounds_step(
        &self,
        dim: usize,
        bounds: &Bounds1,
        lower_aff: &Affine2,
        upper_aff: &Affine2,
    ) -> (Bounds1, (Affine2, Affine2)) {
        deep_poly_steprelu(dim, bounds.clone(), lower_aff.clone(), upper_aff.clone())
    }

    fn get_activation_pattern(&self, state: &Array2<NNVFloat>) -> Option<Array2<bool>> {
        // This should only be Some in an activation layer (e.g. ReLU)
        Some(state.mapv(|x| x >= 0.0))
    }

    fn forward_star(
        &self,
        star: &Star2,
        dim: Option<usize>,
        input_bounds: Option<Bounds1>,
        parent_bounds: Option<Bounds1>,
    ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool) {
        let dim = dim.unwrap();
        let child_stars = star.step_relu2(dim, &input_bounds);
        let parent_bounds = parent_bounds.unwrap();
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
            let mut bounds = parent_bounds.clone();
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

    fn construct_starnodetype(&self, child_ids: &Vec<usize>, dim: Option<usize>) -> StarNodeType {
        debug_assert_gt!(child_ids.len(), 0);
        StarNodeType::StepRelu {
            dim: dim.unwrap(),
            fst_child_idx: child_ids[0],
            snd_child_idx: child_ids.get(1).copied(),
        }
    }
}

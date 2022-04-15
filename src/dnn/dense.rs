use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::graph::Operation;
use crate::star::Star2;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Deref;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Dense {
    aff: Affine2,
}

impl Dense {
    pub const fn new(aff: Affine2) -> Self {
        Self { aff }
    }

    pub fn from_parts(mul: Array2<NNVFloat>, add: Array1<NNVFloat>) -> Self {
        Self {
            aff: Affine2::new(mul, add),
        }
    }
}

impl Operation for Dense {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_shapes(&self) -> Vec<TensorShape> {
        vec![TensorShape::new(vec![Some(self.aff.input_dim())])]
    }

    fn output_shapes(&self) -> Vec<TensorShape> {
        vec![TensorShape::new(vec![Some(self.aff.output_dim())])]
    }
    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        debug_assert_eq!(input.len(), 1);
        debug_assert_eq!(input[0].ndim(), 1);
        vec![self.aff.apply(&input[0].view())]
    }

    fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
        vec![self.aff.apply_matrix(&input[0].view())]
    }

    fn apply_bounds(
        &self,
        bounds: &[&Bounds1],
        lower_aff: &[&Affine2],
        upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        assert_eq!(1, bounds.len());
        let new_lower = self.aff.signed_compose(&lower_aff[0], &upper_aff[0]);
        let new_upper = self.aff.signed_compose(&upper_aff[0], &lower_aff[0]);
        vec![(self.aff.signed_apply(&bounds[0]), new_lower, new_upper)]
    }

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        stars: Vec<StarRef>,
        _activation_idx: Option<usize>,
        _input_bounds: &Bounds1,
        parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>> {
        assert_eq!(stars.len(), 1);
        assert!(parent_local_output_bounds_opt.map_or(true, |b| b.len() == 1));
        vec![vec![(stars[0].affine_map2(&self.aff), None)]]
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dense {}", self.aff.output_dim())
    }
}

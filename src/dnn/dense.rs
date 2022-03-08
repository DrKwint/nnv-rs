use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::dnn::layer::Layer;
use crate::star::Star2;
use crate::star_node::StarNodeType;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;

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

#[typetag::serde]
impl Layer for Dense {
    fn input_shape(&self) -> TensorShape {
        TensorShape::new(vec![Some(self.aff.input_dim())])
    }

    fn output_shape(&self) -> TensorShape {
        TensorShape::new(vec![Some(self.aff.output_dim())])
    }
    fn forward1(&self, input: &Array1<NNVFloat>) -> Array1<NNVFloat> {
        debug_assert_eq!(input.ndim(), 1);
        self.aff.apply(&input.view())
    }

    fn forward2(&self, input: &Array2<NNVFloat>) -> Array2<NNVFloat> {
        self.aff.apply_matrix(&input.view())
    }

    fn apply_bounds(
        &self,
        bounds: &Bounds1,
        lower_aff: &Affine2,
        upper_aff: &Affine2,
    ) -> (Bounds1, (Affine2, Affine2)) {
        let new_lower = self.aff.signed_compose(lower_aff, upper_aff);
        let new_upper = self.aff.signed_compose(upper_aff, lower_aff);
        (self.aff.signed_apply(bounds), (new_lower, new_upper))
    }

    fn forward_star(
        &self,
        star: &Star2,
        _activation_idx: Option<usize>,
        _input_bounds: Option<Bounds1>,
        _parent_bounds: Option<Bounds1>,
    ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool) {
        (vec![star.affine_map2(&self.aff)], vec![None], false)
    }

    fn construct_starnodetype(&self, child_ids: &[usize], _dim: Option<usize>) -> StarNodeType {
        debug_assert_eq!(child_ids.len(), 1);
        StarNodeType::Affine {
            child_idx: child_ids[0],
        }
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dense {}", self.aff.output_dim())
    }
}

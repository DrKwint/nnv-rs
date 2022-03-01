use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::star::Star2;
use crate::star_node::StarNodeType;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::{Array1, Array2};
use std::fmt::{Debug, Display};

#[typetag::serde(tag = "type")]
pub trait Layer: Display + Debug {
    fn input_shape(&self) -> TensorShape {
        panic!()
    }
    fn output_shape(&self) -> TensorShape {
        panic!()
    }

    fn forward1(&self, input: &Array1<NNVFloat>) -> Array1<NNVFloat>;
    fn forward2(&self, input: &Array2<NNVFloat>) -> Array2<NNVFloat>;
    fn apply_bounds(
        &self,
        bounds: &Bounds1,
        lower_aff: &Affine2,
        upper_aff: &Affine2,
    ) -> (Bounds1, (Affine2, Affine2));
    fn apply_bounds_step(
        &self,
        _dim: usize,
        _bounds: &Bounds1,
        _lower_aff: &Affine2,
        _upper_aff: &Affine2,
    ) -> (Bounds1, (Affine2, Affine2)) {
        panic!();
    }

    /// Returns the set of children stars with their input_bounds.
    /// In the case that there is one, sets the bool to whether the output bounds can be copied.
    fn forward_star(
        &self,
        star: &Star2,
        activation_idx: Option<usize>,
        input_bounds: Option<Bounds1>,
        parent_bounds: Option<Bounds1>,
    ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool);
    fn construct_starnodetype(&self, child_ids: &Vec<usize>, dim: Option<usize>) -> StarNodeType;

    fn input_dims(&self) -> usize {
        self.input_shape().dims()
    }

    fn output_dims(&self) -> usize {
        self.input_shape().dims()
    }

    fn is_activation(&self) -> bool {
        // This should be implemented in activation layers to return true
        false
    }

    fn get_activation_pattern(&self, _state: &Array2<NNVFloat>) -> Option<Array2<bool>> {
        // This should only be Some in an activation layer (e.g. ReLU)
        None
    }
}

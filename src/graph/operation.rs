use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::star::Star2;
use crate::star_node::StarNodeType;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use dyn_clone::DynClone;
use ndarray::{Array1, Array2};
use std::any::Any;
use std::fmt::{Debug, Display};

/// Operations may not be stateful. I.e., they must deterministically produce identical outputs from identical inputs.
/// State may be simulated with additional inputs/outputs and with a steppable operation. Further, the number of outputs
/// from a step operation must be equal to the number of outputs from the non-stepped version of the operation.
#[typetag::serde(tag = "type")]
pub(crate) trait Operation: DynClone + Display + Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn num_steps(&self) -> Option<usize> {
        None
    }

    fn input_shapes(&self) -> Vec<TensorShape> {
        panic!()
    }

    fn output_shapes(&self) -> Vec<TensorShape> {
        panic!()
    }

    fn forward1(&self, input: &Vec<Array1<NNVFloat>>) -> Vec<Array1<NNVFloat>>;
    fn forward2(&self, input: &Vec<Array2<NNVFloat>>) -> Vec<Array2<NNVFloat>>;
    fn apply_bounds(
        &self,
        bounds: &Vec<Bounds1>,
        lower_aff: &Vec<Affine2>,
        upper_aff: &Vec<Affine2>,
    ) -> Vec<(Bounds1, Affine2, Affine2)>;
    fn apply_bounds_step(
        &self,
        _dim: usize,
        _bounds: &Vec<Bounds1>,
        _lower_aff: &Vec<Affine2>,
        _upper_aff: &Vec<Affine2>,
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        panic!();
    }

    /// Returns the set of children stars with their input_bounds.
    /// In the case that there is one, sets the bool to whether the output bounds can be copied.
    fn forward_star(
        &self,
        star: &Star2,
        activation_idx: Option<usize>,
        input_bounds: Option<Vec<Bounds1>>,
        parent_bounds: Option<Vec<Bounds1>>,
    ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool);
    fn construct_starnodetype(&self, child_ids: &[usize], dim: Option<usize>) -> StarNodeType;

    fn inputs_dims(&self) -> Vec<usize> {
        self.input_shapes()
            .into_iter()
            .map(|input| input.dims())
            .collect()
    }

    fn outputs_dims(&self) -> Vec<usize> {
        self.input_shapes()
            .into_iter()
            .map(|output| output.dims())
            .collect()
    }

    fn is_activation(&self) -> bool {
        // This should be implemented in activation layers to return true
        false
    }

    fn get_activation_pattern(&self, _state: Vec<&Array2<NNVFloat>>) -> Option<Vec<Array2<bool>>> {
        // This should only be Some in an activation layer (e.g. ReLU)
        None
    }
}

// This implements `Clone` for the trait
dyn_clone::clone_trait_object!(Operation);

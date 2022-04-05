use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::star::Star2;
// use crate::star::Star2;
use crate::dnn::{Conv, Dense, Interpolate, ReLU};
use crate::graph::PhysicalOp;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use enum_dispatch::enum_dispatch;
use ndarray::{Array1, Array2};
use std::any::Any;
use std::fmt::{Debug, Display};
use std::ops::Deref;

#[cfg(test)]
use crate::test_graphs::DummyOperation;
#[cfg(test)]
use crate::test_graphs::{SimpleAdd, SimpleMultiply, SimpleSquare};

/// Operations may not be stateful. I.e., they must deterministically produce identical outputs from identical inputs.
/// State may be simulated with additional inputs/outputs and with a steppable operation. Further, the number of outputs
/// from a step operation must be equal to the number of outputs from the non-stepped version of the operation.
#[enum_dispatch]
pub trait Operation: Clone + Debug + Send + Sync {
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

    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>>;
    fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>>;
    fn apply_bounds(
        &self,
        bounds: &[Bounds1],
        lower_aff: &[Affine2],
        upper_aff: &[Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)>;
    fn apply_bounds_step(
        &self,
        _dim: usize,
        _bounds: &[Bounds1],
        _lower_aff: &[Affine2],
        _upper_aff: &[Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        panic!();
    }

    /// Returns the set of children stars with their input_bounds.
    /// In the case that there is one, sets the bool to whether the output bounds can be copied.
    ///
    /// We pass axis_aligned_input_bounds through each operation because it's very cheap to update and expensive to calculate.
    ///
    /// # Arguments
    ///
    /// * `parent_stars` - The stars used as input to the operation.
    /// * `step_id` - The (optional) step of the operation.
    /// * `axis_aligned_input_bounds` - Optional outer bounds on the entire DNN's input set, must be passed if it's defined on the StarSet
    ///
    /// # Returns
    ///
    /// * `child_stars` -
    /// * `children_axis_aligned_input_bounds` -
    /// * `same_output_bounds` - Whether the children have the same output bounds as the parents. See assumptions above.
    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        parent_stars: Vec<StarRef>,
        step_id: Option<usize>,
        parent_axis_aligned_input_bounds: Vec<Bounds1Ref>,
    ) -> (Vec<Star2>, Vec<Bounds1>, bool);

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

    fn get_activation_pattern(&self, _state: &[&Array2<NNVFloat>]) -> Option<Vec<Array2<bool>>> {
        // This should only be Some in an activation layer (e.g. ReLU)
        None
    }
}

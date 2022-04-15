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
use std::fmt::Debug;
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

    /// Propogates the bounds and transforms the lower and upper constraints according to the operation
    ///
    /// # Description
    ///
    /// DeepPoly approximates output bounds of a network with respect to some input bounds by propagating
    /// constraints through the network. These constraints are represented in the lower and upper affines,
    /// i.e. lower and upper bounding linear equations. Each operation transforms these constraints in some
    /// way and returns the new constraints along with concrete local output bounds.
    ///
    /// # Inputs
    ///
    /// * `bounds` - The local output bounds of each input to the operation
    /// * `lower_aff` - The linear constraints the bound the input set from below.
    /// * `upper_aff` - The linear constraints the bound the input set from above.
    ///
    /// # Returns
    ///
    /// For each output of the operation, returns the tuple:
    /// * `bounds` - The new local output bounds of each input to the operation
    /// * `lower_aff` - The new linear constraints the bound the input set from below.
    /// * `upper_aff` - The new linear constraints the bound the input set from above.
    fn apply_bounds(
        &self,
        bounds: &[&Bounds1],
        lower_aff: &[&Affine2],
        upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)>;

    /// Propogates the bounds and transforms the constraints just as in `apply_bounds`, except on a single
    /// step of the operation.
    fn apply_bounds_step(
        &self,
        _dim: usize,
        _bounds: &[&Bounds1],
        _lower_aff: &[&Affine2],
        _upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        panic!();
    }

    /// Calculates output stars from an operation along with the necessary input stars.
    ///
    /// # Description
    ///
    /// Returns the set of children stars with their input_bounds, one tuple for each output of the operation.
    /// In the case that there is one, sets the bool to whether the output bounds can be copied.
    ///
    /// We pass axis_aligned_input_bounds through each operation because it's very cheap to update and expensive to calculate.
    ///
    /// # Arguments
    ///
    /// * `parent_stars` - The stars used as input to the operation.
    /// * `step_id` - The (optional) step of the operation.
    /// * `input_bounds` - The input bounds to the starset.
    /// * `parent_local_output_bounds_opt` - The local output bounds of the parent nodes.
    ///
    /// # Returns
    ///
    /// For each `output_representation_id`, a vector of tuples is returned of:
    /// * `child_stars` - The stars for the `output_representation_id`
    /// * `child_local_output_bounds` - The local output bounds of the child if they are simple to calculate
    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        parent_stars: Vec<StarRef>,
        step_id: Option<usize>,
        input_bounds: &Bounds1,
        parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>>;

    fn inputs_dims(&self) -> Vec<usize> {
        self.input_shapes()
            .into_iter()
            .map(|input| input.dims())
            .collect()
    }

    fn outputs_dims(&self) -> Vec<usize> {
        self.output_shapes()
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

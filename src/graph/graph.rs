use super::execute_engine::ExecuteError;
use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::star::Star2;
use crate::star_node::StarNodeType;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use dyn_clone::DynClone;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Display};

pub type RepresentationId = usize;
pub type OperationId = usize;

#[derive(Debug)]
pub enum GraphError {
    GenericError,
    AnotherOpProducesOutput,
}

#[typetag::serde(tag = "type")]
pub trait Operation: DynClone + Display + Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn num_steps(&self) -> Option<usize>;

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
    fn construct_starnodetype(&self, child_ids: &[usize], dim: Option<usize>) -> StarNodeType;

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

// This implements `Clone` for the trait
dyn_clone::clone_trait_object!(Operation);

/// Each RepresentationId is created uniquely by a single OperationNode
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationNode {
    operation: Box<dyn Operation>,
    inputs: Vec<RepresentationId>,
    outputs: Vec<RepresentationId>,
}

impl OperationNode {
    pub fn new(
        operation: Box<dyn Operation>,
        inputs: Vec<RepresentationId>,
        outputs: Vec<RepresentationId>,
    ) -> Self {
        Self {
            operation,
            inputs,
            outputs,
        }
    }

    pub fn get_operation(&self) -> &Box<dyn Operation> {
        &self.operation
    }

    pub fn get_input_ids(&self) -> &Vec<RepresentationId> {
        &self.inputs
    }

    pub fn get_output_ids(&self) -> &Vec<RepresentationId> {
        &self.outputs
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    representation_ops: HashMap<RepresentationId, OperationId>, // representation to idx of HashMap
    operation_nodes: Vec<OperationNode>,                        // topo sorted list of operations
}

impl Graph {
    /// Get the specific id of the operation that produces a specific representation
    ///
    /// # Arguments
    ///
    /// * `id` - Representation whose producing operation we are trying to retrieve
    pub fn get_representation_op_id(&self, id: RepresentationId) -> Option<OperationId> {
        self.representation_ops.get(&id).cloned()
    }

    pub fn get_operation_node(&self, id: OperationId) -> Option<&OperationNode> {
        self.operation_nodes.get(id)
    }

    pub fn add_operation(
        &mut self,
        op: Box<dyn Operation>,
        inputs: Vec<RepresentationId>,
        outputs: Vec<RepresentationId>,
    ) -> Result<OperationId, GraphError> {
        let node = OperationNode::new(op, inputs, outputs);
        self.add_operation_node(node)
    }

    /// Add an operation node to the graph. Nodes should be added in a topological order
    ///
    /// # Arguments
    ///
    /// * `node`: The node to add
    pub fn add_operation_node(&mut self, node: OperationNode) -> Result<OperationId, GraphError> {
        let node_id = self.operation_nodes.len();

        if node
            .get_output_ids()
            .iter()
            .map(|&id| self.representation_ops.insert(id, node_id))
            .any(|x| x.is_some())
        {
            return Err(GraphError::AnotherOpProducesOutput);
        }

        self.operation_nodes.push(node);

        Ok(node_id)
    }
}

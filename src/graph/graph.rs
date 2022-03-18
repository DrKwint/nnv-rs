use super::operation::Operation;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Unique key for an operation scoped to a Graph.
pub type OperationId = usize;

/// # Description
///
/// Unique key for a representation scoped to a Graph. I.e., something that is input/output of the graph ops.
/// E.g., tensors, stars, bounds.
#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize, Hash)]
pub struct RepresentationId {
    pub representation_node_id: usize, // usize used as unique key scoped to Graph
    /// If this representation is internal to a stepped operation (i.e. is produced and consumed by that operation),
    /// This field encodes the index of the step run to create this representation if this representation is intermediate,
    /// and should be None on the final operation output
    pub operation_step: Option<usize>,
}

impl RepresentationId {
    pub fn new(representation_node_id: usize, operation_step: Option<usize>) -> Self {
        Self {
            representation_node_id,
            operation_step,
        }
    }
}

/// # Invariants:
/// - Graph state has all the alive representations required to compute the output representations
pub struct GraphState<T> {
    output_representation_ids: Vec<RepresentationId>,
    alive_representations: HashMap<RepresentationId, T>, // All representations required to run the remaining operations
}

impl<T> GraphState<T> {
    pub fn new(
        output_ids: Vec<RepresentationId>,
        input_representations: HashMap<RepresentationId, T>,
        graph: &Graph,
    ) -> Self {
        // Use the graph to ensure we construct a valid GraphState, i.e., alive representations are
        // enough to calculate the output representations
        todo!()
    }
}

/// A topo-sorted list of operations that transforms representations into representations.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    representation_ops: HashMap<RepresentationId, OperationId>, // OperationId is the one that produces the RepresentationId
    operation_nodes: Vec<OperationNode>,                        // topo sorted list of operations
}

impl Graph {
    /// Get the specific id of the operation that produces a specific representation
    ///
    /// # Arguments
    ///
    /// * `id` - Representation whose producing operation we are trying to retrieve
    pub fn get_representation_op_id(&self, id: &RepresentationId) -> Option<OperationId> {
        self.representation_ops.get(&id).cloned()
    }

    /// Get the ids of the operations that `id` feeds into
    pub fn get_representation_input_op_ids(&self, id: &RepresentationId) -> Vec<OperationId> {
        self.operation_nodes
            .iter()
            .enumerate()
            .filter(|(idx, op_node)| op_node.get_input_ids().contains(id))
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>()
    }

    pub fn get_operation_node(&self, id: &OperationId) -> Option<&OperationNode> {
        self.operation_nodes.get(*id)
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

#[derive(Debug)]
pub enum GraphError {
    GenericError,
    AnotherOpProducesOutput,
}

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

use super::operation::Operation;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// Unique key for an operation scoped to a Graph.
pub type OperationId = usize;

/// # Description
///
/// Unique key for a representation scoped to a Graph. I.e., something that is input/output of the graph ops.
/// E.g., tensors, stars, bounds.
#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize, Hash)]
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
/// - Graph state has all the alive representations required to compute the output representations, including the output representations
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GraphState<T: Debug + Clone> {
    /// The representations being calculated
    pub output_representation_ids: Vec<RepresentationId>,

    /// All representations required to run the remaining operations, including final output representations
    pub alive_representations: HashMap<RepresentationId, T>,

    /// The number of operations that require each alive representations (`output_representation_ids` count)
    pub reference_counts: HashMap<RepresentationId, usize>,
}

impl<T: Debug + Clone> GraphState<T> {
    pub fn new(
        output_ids: Vec<RepresentationId>,
        input_representations: HashMap<RepresentationId, T>,
        graph: &Graph,
    ) -> Self {
        // TODO: Use the graph to ensure we construct a valid GraphState, i.e., alive representations are
        // enough to calculate the output representations

        let input_representation_ids = input_representations
            .iter()
            .map(|(&repr_id, _)| repr_id)
            .collect::<Vec<_>>();

        let reference_counts = graph.get_reference_counts(&output_ids, &input_representation_ids);

        Self {
            output_representation_ids: output_ids,
            alive_representations: input_representations,
            reference_counts,
        }
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

    /// Get the ids of the operations that `id` feeds into.
    ///
    /// # Description
    ///
    /// If an `id` has no `operation_step`, then the operation that has `id` as an input representation id is returned.
    /// If `id` has an `operation_step`, then *the* operation that produces an identical id with no operation step is returned.
    pub fn get_representation_input_op_ids(&self, id: &RepresentationId) -> Vec<OperationId> {
        if id.operation_step.is_none() {
            self.operation_nodes
                .iter()
                .enumerate()
                .filter(|(_, op_node)| op_node.get_input_ids().contains(id))
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        } else {
            let mut id_no_step = id.clone();
            id_no_step.operation_step = None;
            self.operation_nodes
                .iter()
                .enumerate()
                .filter(|(_, op_node)| op_node.get_output_ids().contains(&id_no_step))
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        }
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

        if let Some(op_id) = node
            .get_output_ids()
            .iter()
            .map(|&id| self.representation_ops.insert(id, node_id))
            .filter(|&old_id| old_id.is_some())
            .map(|old_id| old_id.unwrap())
            .next()
        {
            return Err(GraphError::AnotherOpProducesOutput { op_id });
        }

        self.operation_nodes.push(node);

        Ok(node_id)
    }

    /// Given a `GraphState`, finds the next operation to perform in the graph
    pub fn get_next_operation<T: Clone + Debug>(
        &self,
        state: &GraphState<T>,
    ) -> (OperationId, Option<usize>) {
        // 1. Collect all operations that take the alive representations as input
        let possible_ops = state
            .alive_representations
            .iter()
            .map(|(repr_id, _)| {
                self.get_representation_input_op_ids(repr_id)
                    .into_iter()
                    .map(|op_id| (op_id, *repr_id))
            })
            .flatten()
            .collect::<Vec<_>>();

        // 2. Find the next operation in the topo-sorted subgraph
        let mut sorted_ops = possible_ops.into_iter().collect::<Vec<_>>();
        sorted_ops.sort_by(|a, b| a.0.cmp(&b.0));
        let &(op, repr_id) = sorted_ops.first().unwrap();

        (op, repr_id.operation_step)
    }

    ///  Gets the reference counts of the input_ids
    ///
    /// # Description
    ///
    /// Calculates the number of operations that the inputs feed in the subgraph that produces the
    /// outputs. This includes the trivial operation if an output id = input id.
    ///
    /// This operation is expensive as it needs to calculate the subgraph and should be run as few
    /// times as necessary.
    pub fn get_reference_counts(
        &self,
        output_ids: &[RepresentationId],
        input_ids: &[RepresentationId],
    ) -> HashMap<RepresentationId, usize> {
        let mut reference_counts: HashMap<RepresentationId, usize> =
            HashMap::from_iter(input_ids.into_iter().map(|&repr_id| (repr_id, 0 as usize)));

        let operation_set = self.get_operation_set(output_ids, input_ids).unwrap();

        output_ids.iter().for_each(|out_id| {
            if let Some(ref_count) = reference_counts.get_mut(out_id) {
                *ref_count += 1;
            }
        });

        operation_set.iter().for_each(|op_id| {
            let op_node = self.get_operation_node(op_id).unwrap();

            let mut step_inputs_found = false;

            // Check if stepped outputs for the op node are given
            op_node.get_output_ids().iter().for_each(|out_id| {
                let mut step_ids = input_ids
                    .iter()
                    .filter(|&repr_id| {
                        repr_id.representation_node_id == out_id.representation_node_id
                    })
                    .collect::<Vec<_>>();
                if !step_ids.is_empty() {
                    step_inputs_found = true;
                    step_ids
                        .sort_by(|a, b| a.operation_step.unwrap().cmp(&b.operation_step.unwrap()));
                    *reference_counts.get_mut(*step_ids.last().unwrap()).unwrap() += 1;
                }
            });

            if step_inputs_found {
                return;
            }

            op_node.get_input_ids().iter().for_each(|in_id| {
                if let Some(ref_count) = reference_counts.get_mut(in_id) {
                    *ref_count += 1;
                }
            });
        });

        reference_counts
    }

    /// Calculates a subgraph of operations necessary to compute the outputs from the inputs
    pub fn get_operation_set(
        &self,
        output_ids: &[RepresentationId],
        input_ids: &[RepresentationId],
    ) -> Result<HashSet<OperationId>, GraphError> {
        // Set of representations that still need to be calculated
        let mut active_representation_ids = output_ids.iter().collect::<Vec<_>>();

        // set of ops indices required to produce the outputs
        let mut op_node_set: HashSet<OperationId> = HashSet::new();

        // Set of representations that have already been calculated by some previous op
        let mut finished_representations: HashSet<RepresentationId> =
            HashSet::from_iter(input_ids.iter().map(|&x| x));

        while !active_representation_ids.is_empty() {
            // Get next representation we need
            let active_repr_id = active_representation_ids
                .pop()
                .ok_or(GraphError::PoppedEmptyStack)?;

            // The active id may already be added to finished representations if the producing
            // operation has more than one output.
            if finished_representations.contains(&active_repr_id) {
                continue;
            }

            // If not, try to get it from an operation
            // `op_id` is the id of the operation that produces `active_repr_id`
            let op_id = self.get_representation_op_id(&active_repr_id).ok_or(
                GraphError::NoOpCreatesRepresentation {
                    repr_id: *active_repr_id,
                },
            )?;

            let op_node = self
                .get_operation_node(&op_id)
                .ok_or(GraphError::OperationNotExist { op_id })?;

            // 1. Input representation is stepped
            // 2. Output representation is stepped
            // RepresentationId { node_id: _, step: Some(_) } is a stepped representation
            // 3. Running is critical to transforming a representation (e.g., building a starset tree)

            op_node_set.insert(op_id);
            // Handle op outputs
            if op_node
                .get_output_ids()
                .iter()
                .map(|x| finished_representations.insert(*x))
                .any(|x| !x)
            {
                return Err(GraphError::AnotherOpProducesOutput { op_id });
            }

            // Handle op inputs
            op_node.get_input_ids().iter().for_each(|input_id| {
                if !finished_representations.contains(input_id) {
                    active_representation_ids.push(input_id);
                }
            });
        }
        Ok(op_node_set)
    }
}

#[derive(Debug)]
pub enum GraphError {
    GenericError,
    PoppedEmptyStack,
    OperationNotExist { op_id: OperationId },
    NoOpCreatesRepresentation { repr_id: RepresentationId },
    AnotherOpProducesOutput { op_id: OperationId },
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

#[cfg(test)]
mod test {
    use crate::test_util::*;
    use proptest::*;

    proptest! {
        #[test]
        fn test_from_sequential(_ in fc_dnn(4, 4, 4, 4)) {
        }
    }
}

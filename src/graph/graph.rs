use crate::dnn::conv::Conv;
use crate::dnn::dense::Dense;
use crate::dnn::interpolate::Interpolate;
use crate::dnn::relu::ReLU;
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};

#[cfg(test)]
use crate::test_graphs::DummyOperation;
#[cfg(test)]
use crate::test_graphs::{SimpleAdd, SimpleMultiply, SimpleSquare};

/// Unique key for an operation scoped to a Graph.
pub type OperationId = usize;

/// # Description
///
/// Unique key for a representation scoped to a Graph. I.e., something that is input/output of the graph ops.
/// E.g., tensors, stars, bounds.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct RepresentationId {
    pub representation_node_id: usize, // usize used as unique key scoped to Graph
    /// If this representation is internal to a stepped operation (i.e. is produced and consumed by that operation),
    /// This field encodes the index of the step run to create this representation if this representation is intermediate,
    /// and should be None on the final operation output
    pub operation_step: Option<usize>,
}

impl RepresentationId {
    pub const fn new(representation_node_id: usize, operation_step: Option<usize>) -> Self {
        Self {
            representation_node_id,
            operation_step,
        }
    }

    #[must_use]
    pub const fn with_step(mut self, operation_step: Option<usize>) -> Self {
        self.operation_step = operation_step;
        self
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
#[derive(Default, Clone)]
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
        if id.operation_step.is_some() {
            self.representation_ops.get(&(*id).with_step(None)).copied()
        } else {
            self.representation_ops.get(id).copied()
        }
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
            let mut id_no_step = *id;
            id_no_step.operation_step = None;
            self.operation_nodes
                .iter()
                .enumerate()
                .filter(|(_, op_node)| op_node.get_output_ids().contains(&id_no_step))
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        }
    }

    pub fn get_operations(&self) -> &Vec<OperationNode> {
        &self.operation_nodes
    }

    pub fn get_operation_node(&self, id: &OperationId) -> Option<&OperationNode> {
        self.operation_nodes.get(*id)
    }

    /// # Errors
    /// TODO
    pub fn add_operation(
        &mut self,
        op: PhysicalOp,
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
    ///
    /// # Errors
    /// TODO
    pub fn add_operation_node(&mut self, node: OperationNode) -> Result<OperationId, GraphError> {
        let node_id = self.operation_nodes.len();

        if let Some(op_id) = node
            .get_output_ids()
            .iter()
            .map(|&id| self.representation_ops.insert(id, node_id))
            .filter(|&old_id| old_id.is_some())
            .map(std::option::Option::unwrap)
            .next()
        {
            return Err(GraphError::AnotherOpProducesOutput { op_id });
        }

        self.operation_nodes.push(node);

        Ok(node_id)
    }

    /// Given a `GraphState`, finds the next operation to perform in the graph
    ///
    /// # Panics
    pub fn get_next_operation<T: Clone + Debug>(
        &self,
        state: &GraphState<T>,
    ) -> (OperationId, Option<usize>) {
        // 1. Collect all operations that take the alive representations as input
        let possible_ops = state.alive_representations.iter().flat_map(|(repr_id, _)| {
            self.get_representation_input_op_ids(repr_id)
                .into_iter()
                .map(|op_id| (op_id, *repr_id))
        });

        // 2. Find the next operation in the topo-sorted subgraph
        let mut sorted_ops = possible_ops.collect::<Vec<_>>();
        sorted_ops.sort_by(|a, b| a.0.cmp(&b.0));
        let &(op, repr_id) = sorted_ops.first().unwrap();

        (op, repr_id.operation_step)
    }

    ///  Gets the reference counts of the `input_ids`
    ///
    /// # Description
    ///
    /// Calculates the number of operations that the inputs feed in the subgraph that produces the
    /// outputs. This includes the trivial operation if an output id = input id.
    ///
    /// This operation is expensive as it needs to calculate the subgraph and should be run as few
    /// times as necessary.
    ///
    /// # Panics
    pub fn get_reference_counts(
        &self,
        output_ids: &[RepresentationId],
        input_ids: &[RepresentationId],
    ) -> HashMap<RepresentationId, usize> {
        let mut reference_counts: HashMap<RepresentationId, usize> = input_ids
            .iter()
            .map(|&repr_id| (repr_id, 0_usize))
            .collect();

        let operation_set = self.get_operation_set(output_ids, input_ids).unwrap();

        for out_id in output_ids {
            if let Some(ref_count) = reference_counts.get_mut(out_id) {
                *ref_count += 1;
            }
        }

        for op_id in &operation_set {
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
                continue;
            }

            op_node.get_input_ids().iter().for_each(|in_id| {
                if let Some(ref_count) = reference_counts.get_mut(in_id) {
                    *ref_count += 1;
                }
            });
        }

        reference_counts
    }

    /// Calculates a subgraph of operations necessary to compute the outputs from the inputs
    ///
    /// # Description
    ///
    /// Computes a subgraph of necessary operations to calculate output representations from
    /// inputs ones. The most general case is that where steps of operations are not considered.
    /// In this case, we compute via search the necessary operations to produce outputs.
    ///
    /// At each step, we:
    /// 1. Get the next operation that needs to be computed by popping off the
    ///    `active_representation_ids` set.
    /// 2. Check to ensure the operation still needs to be computed by checking the
    ///    `finished_representations` set. If not, we continue to the next representation
    /// 3. Get the operation that produces the next representation in the stack.
    /// 4. We check for inputs to the operation that have not yet been computed, i.e. are not in
    ///    the `finished_representations` set. We add these representations to the stack, i.e.
    ///    `active_representation_ids`.
    /// 5. We add the outputs of the operation to the `finished_representations` set. Note that we
    ///    may add multiple representations at this step at this is why step (2.) is required.
    ///
    /// At initialization, the `inputs_ids` are added to the `finished_representations` set and the
    /// `output_ids` are added to the `active_representation_ids` set.
    ///
    /// ## Subgraph calculation with operation steps
    ///
    /// The more difficult case is when stepped representations are provied, i.e. when some calculation
    /// has been done for an operation, but not yet all of it. To handle this we have 3 additional
    /// cases to consider:
    /// 1. An operation node has stepped inputs, but not stepped outputs.
    /// 2. An operation node has stepped outputs, but not stepped inputs.
    /// 3. An operation node has stepped inputs and stepped outputs.
    ///
    /// The first and second cases are simple to deal with, with a few assumptions. The first assumption
    /// is that for a stepped operation with multiple outputs, all outputs are given for the same step
    /// and further, only one step of the operation may be input (however multiple output steps may be
    /// specified). Given this set of assumptions, for both cases, we simply run the producing operation
    /// to the end. While this does incur some additional cost with unnecessary steps being calculated,
    /// we believe this cost to be low, due to the decomposability and cheapness of stepped operations.
    ///
    /// Finally, for the third case we can test for this case explicitly. For such a case, we can add the
    /// operation node as a required operation and then proceed to treat the output of the node as a whole
    /// as an input to the rest of the algorithm, removing the output id from the calculation. Incidentally,
    /// accounting for case 1 also accounts for the inputs of case 3, so we just need to make sure we don't
    /// have double outputs when constructing `active_representation_ids`.
    ///
    /// # Errors
    /// # Panics
    pub fn get_operation_set(
        &self,
        output_ids: &[RepresentationId],
        input_ids: &[RepresentationId],
    ) -> Result<HashSet<OperationId>, GraphError> {
        // set of ops indices required to produce the outputs
        let mut op_node_set: HashSet<OperationId> = HashSet::new();

        // Set of representations that have already been calculated by some previous op
        let mut finished_representations: HashSet<RepresentationId> =
            input_ids.iter().copied().collect();

        // Test for case 1: input ids are stepped
        input_ids
            .iter()
            .filter(|input_id| input_id.operation_step.is_some())
            .for_each(|input_id| {
                let op_id = self.get_representation_op_id(input_id).unwrap();
                op_node_set.insert(op_id);
                let op_node = self.get_operation_node(&op_id).unwrap();
                op_node.get_output_ids().iter().for_each(|id| {
                    finished_representations.insert(*id);
                });
            });

        // Set of representations that still need to be calculated, checking for case 3
        let mut active_representation_ids: Vec<_> = output_ids
            .iter()
            .filter(|out_id| !finished_representations.contains(&out_id.with_step(None)))
            .collect();

        while !active_representation_ids.is_empty() {
            // Get next representation we need
            let active_repr_id = active_representation_ids
                .pop()
                .ok_or(GraphError::PoppedEmptyStack)?;

            // The active id may already be added to finished representations if the producing
            // operation has more than one output.
            if finished_representations.contains(active_repr_id) {
                continue;
            }

            // If not, try to get it from an operation
            // `op_id` is the id of the operation that produces `active_repr_id`
            let op_id = {
                let op_id_check = self.get_representation_op_id(active_repr_id).ok_or(
                    GraphError::NoOpCreatesRepresentation {
                        repr_id: *active_repr_id,
                    },
                );
                if op_id_check.is_err() {
                    println!("Here");
                }
                op_id_check?
            };

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

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr_ops_str = self
            .representation_ops
            .iter()
            .map(|entry| format!("\t{:?}", entry))
            .join("\n");
        let op_strs = self
            .operation_nodes
            .iter()
            .map(|node| format!("\t{:?}", node))
            .join("\n");

        write!(
            f,
            "Graph {{\nrepresentation_ops: \n{}\noperation_nodes: \n{}\n}}",
            &repr_ops_str, &op_strs
        )
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

#[enum_dispatch(Operation)]
pub enum PhysicalOp {
    Dense,
    Conv,
    ReLU,
    Interpolate,

    #[cfg(test)]
    DummyOperation,
    #[cfg(test)]
    SimpleAdd,
    #[cfg(test)]
    SimpleMultiply,
    #[cfg(test)]
    SimpleSquare,
}

impl Debug for PhysicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(x) => write!(f, "{:?}", x),
            Self::Conv(x) => write!(f, "{:?}", x),
            Self::ReLU(x) => write!(f, "{:?}", x),
            Self::Interpolate(x) => write!(f, "{:?}", x),
            #[cfg(test)]
            Self::DummyOperation(x) => write!(f, "{:?}", x),
            #[cfg(test)]
            Self::SimpleAdd(x) => write!(f, "{:?}", x),
            #[cfg(test)]
            Self::SimpleMultiply(x) => write!(f, "{:?}", x),
            #[cfg(test)]
            Self::SimpleSquare(x) => write!(f, "{:?}", x),
        }
    }
}

impl Clone for PhysicalOp {
    fn clone(&self) -> Self {
        match self {
            Self::Dense(x) => Self::Dense(x.clone()),
            Self::Conv(x) => Self::Conv(x.clone()),
            Self::ReLU(x) => Self::ReLU(x.clone()),
            Self::Interpolate(x) => Self::Interpolate(x.clone()),
            #[cfg(test)]
            Self::DummyOperation(x) => Self::DummyOperation(x.clone()),
            #[cfg(test)]
            Self::SimpleAdd(x) => Self::SimpleAdd(x.clone()),
            #[cfg(test)]
            Self::SimpleMultiply(x) => Self::SimpleMultiply(x.clone()),
            #[cfg(test)]
            Self::SimpleSquare(x) => Self::SimpleSquare(x.clone()),
        }
    }
}

/// Each `RepresentationId` is created uniquely by a single `OperationNode`
#[derive(Debug, Clone)]
pub struct OperationNode {
    operation: PhysicalOp,
    inputs: Vec<RepresentationId>,
    outputs: Vec<RepresentationId>,
}

impl OperationNode {
    pub fn new(
        operation: PhysicalOp,
        inputs: Vec<RepresentationId>,
        outputs: Vec<RepresentationId>,
    ) -> Self {
        Self {
            operation,
            inputs,
            outputs,
        }
    }

    pub fn get_operation(&self) -> &PhysicalOp {
        &self.operation
    }

    pub const fn get_input_ids(&self) -> &Vec<RepresentationId> {
        &self.inputs
    }

    pub const fn get_output_ids(&self) -> &Vec<RepresentationId> {
        &self.outputs
    }
}

#[cfg(test)]
mod test {
    use crate::graph::operation::Operation;
    use crate::test_util::*;
    use proptest::*;

    proptest! {
        #[test]
        fn test_step_set_operations(dnn in fc_dnn(4, 4, 4, 4)) {
            let operation_nodes = dnn.get_graph().get_operations();
            for (op_id, op_node) in operation_nodes.iter().enumerate() {
                if let Some(num_steps) = op_node.get_operation().num_steps() {
                    for start_step in 0..(num_steps - 1) {
                        // Avoids underflow
                        if start_step == num_steps - 1 {
                            continue;
                        }
                        let end_iter = ((start_step+1)..(num_steps-1)).map(|step| Some(step)).chain(vec![None].into_iter());
                        for end_step in end_iter {
                            let input_id = op_node.get_output_ids()[0].clone().with_step(Some(start_step));
                            let output_id = op_node.get_output_ids()[0].clone().with_step(end_step);
                            let stepped_nodes = dnn.get_graph().get_operation_set(&vec![output_id], &vec![input_id]);
                            prop_assert!(stepped_nodes.is_ok(), "input: {:?} output: {:?} err: {:?}", input_id, output_id, stepped_nodes);
                            let stepped_nodes: Vec<_> = stepped_nodes.unwrap().into_iter().collect();
                            prop_assert_eq!(stepped_nodes.len(), 1);
                            prop_assert_eq!(stepped_nodes[0], op_id);
                        }
                    }
                }
            }
        }
    }
}

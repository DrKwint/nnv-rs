//! ## Engine lifecycle
//! 1. Create the `Engine` by passing it a `Graph`
//! 2. Call `run` or a variant to transform input representations to output representations
//! 3. GOTO 2
//!
//! Calling `run` requires a closure or an `OperationVisitor` trait object. This visitor will use
//! the operation data and input representation data to calculate output representations. It's
//! setup like this to facilitate the use of new representations.
use super::graph::{Graph, GraphState, OperationId, OperationNode, RepresentationId};
use super::operation::Operation;
use super::{GraphError, PhysicalOp};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;

pub struct Engine<'a> {
    graph: &'a Graph,
}

impl<'a> Engine<'a> {
    pub const fn new(graph: &'a Graph) -> Self {
        Self { graph }
    }

    /// Calculates output representations of a sub-graph, given the input representations and a visitor.
    ///
    /// # Description
    ///
    /// This function works with both visitors of a node and steps within a node.
    /// For instance, ReLU layers of a network are broken down into a step ReLU at
    /// each dimension. It is assumed that intermediate outputs of node steps take
    /// the same form as nodes themselves.
    ///
    /// # Arguments
    ///
    /// * `output_ids` - Ids of the representations to calculate.
    /// * `inputs` - Set of starting inputs required to calculate outputs
    ///     nodal visit call is replaced with the step visits.
    /// * `visit` - Performs the intermediate calculations at each node.
    ///     * Arguments:
    ///         * `op` - The operation of the visited node
    ///         * `inputs` - The input representations
    ///         * `step` - The step to calculate. `None` is passed in when inputs to the node are given.
    ///     * Returns:
    ///         * `new_step` - The step that was just calculated. Use `None` to signify that the operation is complete
    ///         * `repr` - The representation that was calculated.
    ///
    /// # Returns
    ///
    /// * `outputs` - The outputs for each id in `output_ids`
    ///
    /// # Errors
    /// TODO
    ///
    /// # Panics
    /// TODO
    pub fn run<T: Clone + Debug>(
        &self,
        output_ids: &Vec<RepresentationId>,
        inputs: &[(RepresentationId, T)],
        mut visit: impl FnMut(&PhysicalOp, &Vec<&T>, Option<usize>) -> (Option<usize>, Vec<T>),
    ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
        self.run_nodal(
            output_ids,
            inputs,
            |_, op_node, inputs, step| -> (Option<usize>, Vec<T>) {
                visit(op_node.get_operation(), inputs, step)
            },
        )
    }

    pub fn run_nodal<T: Clone + Debug>(
        &self,
        output_ids: &Vec<RepresentationId>,
        inputs: &[(RepresentationId, T)],
        mut visit: impl FnMut(
            OperationId,
            &OperationNode,
            &Vec<&T>,
            Option<usize>,
        ) -> (Option<usize>, Vec<T>),
    ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
        let mut state = ExecutionState::<T>::default();

        // Calculate subgraph and path of operations to perform
        // 1. Walk back through operations BFS
        let input_ids = inputs.iter().map(|&(id, _)| id).collect::<Vec<_>>();
        let operation_set = self.graph.get_operation_set(output_ids, &input_ids)?;

        // 2. Order set via the graph's topological ordering.
        let mut op_node_vec: Vec<OperationId> = operation_set.into_iter().collect();

        // This ordering is actually just the ascending ordering of OperationIds
        op_node_vec.sort_unstable();

        if inputs
            .iter()
            .map(|(id, v)| state.set_representation(*id, v.clone()))
            .any(|x| x.is_err())
        {
            return Err(ExecuteError::GenericError);
        }

        // 3. Apply the visitor to every operation and step in order
        for op_id in op_node_vec {
            let op_node = self
                .graph
                .get_operation_node(&op_id)
                .ok_or(ExecuteError::OperationNotExist { op_id })?;
            let op = op_node.get_operation();

            // If `step_start[output_ids]` exist, then use them and the step.
            let mut input_ids = op_node
                .get_output_ids()
                .iter()
                .map(|out_id| state.get_step_start(*out_id).copied())
                .collect::<Option<Vec<_>>>()
                .unwrap_or_else(|| op_node.get_input_ids().clone());
            let mut step = input_ids.first().unwrap().operation_step;

            // Loop over steps in the operation until the final output is returned.
            loop {
                // Calculate outputs of (step) operation
                // Representations of each input to `op_node`
                let (new_step, outputs) = {
                    let reprs = input_ids
                        .iter()
                        .map(|&id| state.get_representation(id))
                        .collect::<Option<Vec<&T>>>()
                        .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                            repr_ids: op_node.get_input_ids().clone(),
                        })?;
                    visit(op_id, op_node, &reprs, step)
                };

                if outputs.len() != op_node.get_output_ids().len() {
                    return Err(ExecuteError::IncorrectOutputsFromVisitor {
                        expected: outputs.len(),
                        given: op_node.get_output_ids().len(),
                    });
                }

                // Store outputs
                for (&repr_id, repr) in op_node.get_output_ids().iter().zip(outputs.into_iter()) {
                    let mut repr_id = repr_id;
                    repr_id.operation_step = new_step;
                    state.set_representation(repr_id, repr)?;
                }

                for &repr_id in op_node.get_output_ids() {
                    if let Some(&done_step) = state.get_step_end(repr_id) {
                        // Check if we just skipped a end step in the operation
                        if new_step.map_or(true, |s| s > done_step) {
                            return Err(ExecuteError::SkippedEndStepOfOperation {
                                repr_id,
                                done_step,
                            });
                        }

                        // Check if we went past the number of steps in the operation
                        if let Some(new_s) = new_step && new_s >= op.num_steps().unwrap() - 1 {
                            return Err(ExecuteError::NewStepLargerThanNumSteps { new_step: new_s, last_step: op.num_steps().unwrap() - 1 });
                        }
                    }
                }

                // Check if the last representation was given
                if new_step.is_none() {
                    break;
                }

                step = new_step;

                if step.unwrap() == 0 {
                    input_ids = op_node.get_output_ids().clone();
                }
                input_ids = input_ids.into_iter().map(|id| id.with_step(step)).collect();

                // Check if the last representation needed for the outputs was given
                for &repr_id in op_node.get_output_ids() {
                    if let Some(&max_step) = state.get_step_end(repr_id) {
                        if new_step.map_or(false, |s| s == max_step) {
                            break;
                        }
                    }
                }
            }
        }

        // Collect and return output representations
        let outputs = output_ids
            .iter()
            .map(|&id| state.get_representation(id).cloned().map(|r| (id, r)))
            .collect::<Option<Vec<(RepresentationId, T)>>>()
            .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                repr_ids: output_ids.clone(),
            })?;

        Ok(outputs)
    }

    /// # Panics
    /// TODO
    pub fn run_graph_state_to<T: Clone + Debug>(
        &self,
        _state: &GraphState<T>,
        _repr_id: RepresentationId,
    ) -> GraphState<T> {
        todo!()
    }

    /// Calculates the output of a visitor given the inputs
    ///
    /// # Arguments
    ///
    /// * `outputs` - Set of representations to calculate
    /// * `inputs` - Set of starting inputs required to calculate outputs
    /// * `visitor` - Performs the intermediate calculations at each node
    ///
    /// # Panics
    /// TODO
    ///
    /// # Errors
    /// TODO
    pub fn run_node_visitor<T: Clone + Debug>(
        &self,
        _output_ids: &[RepresentationId],
        _inputs: &[(RepresentationId, T)],
        _visitor: &mut dyn OperationVisitor<T>,
    ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
        todo!();
        // self.run_node_visit(output_ids, inputs, |op, inp| visitor.visit(op, inp))
    }

    // pub fn run_node_visit<T: Clone + Debug>(
    //     &self,
    //     output_ids: Vec<RepresentationId>,
    //     inputs: Vec<(RepresentationId, T)>,
    //     visit_node: impl FnMut(&Box<dyn Operation>, Vec<&T>) -> Vec<T>,
    // ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
    //     self.run(output_ids, inputs, false, |op, inputs, _| {
    //         visit_node(op, inputs)
    //     })
    //     .map(|res| res)
    // }
}

#[derive(Debug)]
pub enum ExecuteError {
    GenericError,
    PoppedEmptyStack,
    GraphError {
        err: GraphError,
    },
    IncorrectOutputsFromVisitor {
        expected: usize,
        given: usize,
    },
    NoOpCreatesRepresentation {
        repr_id: RepresentationId,
    },
    OperationAddedTwice {
        op_id: OperationId,
    },
    OperationNotExist {
        op_id: OperationId,
    },
    AnotherOpProducesOutput {
        op_id: OperationId,
    },
    OneOfRepresentationsNotExist {
        repr_ids: Vec<RepresentationId>,
    },
    ReprIdGivenByInputsAndOpState {
        repr_ids: HashSet<RepresentationId>,
    },
    VisitStepsTrueForSteppedInputsOrOutputs,
    SkippedEndStepOfOperation {
        repr_id: RepresentationId,
        done_step: usize,
    },
    NewStepLargerThanNumSteps {
        new_step: usize,
        last_step: usize,
    },
    StateAlreadyHasRepresentation {
        rep_id: RepresentationId,
    },
}

impl From<GraphError> for ExecuteError {
    fn from(err: GraphError) -> Self {
        Self::GraphError { err }
    }
}

pub trait OperationVisitor<T: Clone> {
    /// Visits the operations in set topological order. Will always visit child operations before parent operations.
    ///
    /// # Arguments
    ///
    /// * `operation` - The operation being visited
    /// * `inputs` - Inputs to the operation
    fn visit(&mut self, operation: &PhysicalOp, inputs: Vec<&T>) -> Vec<T>;
}

pub struct ExecutionState<T: Clone> {
    /// Keeps track of all representations currently in memory
    representations: HashMap<RepresentationId, T>,

    /// Keeps track of which representations start at a step instead of directly from input
    step_starts: HashMap<usize, RepresentationId>,

    /// Keeps track of which representation ids signal the end of calculation within the node, i.e., shortcutting the computation.
    /// Maps `representation_node_id` -> `step`
    step_ends: HashMap<usize, usize>,
}

impl<T: Clone> ExecutionState<T> {
    pub fn _new(
        representations: HashMap<RepresentationId, T>,
        step_starts: &[RepresentationId],
        step_ends: &[RepresentationId],
    ) -> Self {
        Self {
            representations,
            step_starts: step_starts
                .iter()
                .map(|&repr_id| (repr_id.representation_node_id, repr_id))
                .collect(),
            step_ends: step_ends
                .iter()
                .map(|repr_id| {
                    (
                        repr_id.representation_node_id,
                        repr_id.operation_step.unwrap(),
                    )
                })
                .collect(),
        }
    }

    pub fn get_step_start(&self, repr_id: RepresentationId) -> Option<&RepresentationId> {
        self.step_starts.get(&repr_id.representation_node_id)
    }

    pub fn get_step_end(&self, repr_id: RepresentationId) -> Option<&usize> {
        self.step_ends.get(&repr_id.representation_node_id)
    }
}

impl<T: Clone> Default for ExecutionState<T> {
    fn default() -> Self {
        Self {
            representations: HashMap::new(),
            step_starts: HashMap::new(),
            step_ends: HashMap::new(),
        }
    }
}

impl<T: Clone> ExecutionState<T> {
    pub fn get_representation(&self, representation_id: RepresentationId) -> Option<&T> {
        self.representations.get(&representation_id)
    }

    pub fn set_representation(
        &mut self,
        representation_id: RepresentationId,
        representation: T,
    ) -> Result<(), ExecuteError> {
        if let std::collections::hash_map::Entry::Vacant(e) =
            self.representations.entry(representation_id)
        {
            e.insert(representation);
            Ok(())
        } else {
            // TODO: Specify error
            Err(ExecuteError::StateAlreadyHasRepresentation {
                rep_id: representation_id,
            })
        }
    }
}

// #[derive(Clone, Debug)]
// pub enum StepState<T: Clone + Debug> {
//     Repr(T),                     // A single representation
//     StepRepr(HashMap<usize, T>), // The representations within an operation
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_execute_fc_dnn(dnn in generic_fc_dnn(4,4,6,6)) {
            let engine = Engine::new(dnn.get_graph());

            prop_assert_eq!(1, dnn.get_output_representation_ids().len());

            let num_steps = dnn.get_graph().get_operations().into_iter().fold(0, |acc, x| acc +
                x.get_operation().num_steps().unwrap_or(0)
            );

            let inputs = dnn.get_input_representation_ids().into_iter().map(|&id| (id, 0 as usize)).collect::<Vec<_>>();
            let res = engine.run(dnn.get_output_representation_ids(), &inputs, |op, inputs, step| -> (Option<usize>, Vec<usize>) {
                if let Some(num_steps) = op.num_steps() {
                    let steps = inputs.into_iter().map(|&&x| x + 1).collect();
                    if let Some(step) = step {
                        if step + 2 == num_steps {
                            (None, steps)
                        } else {
                            (Some(step + 1), steps)
                        }
                    } else {
                        if num_steps == 1 {
                            (None, steps)
                        } else {
                            (Some(0), steps)
                        }
                    }
                } else {
                    (None, inputs.into_iter().map(|&&x| x).collect())
                }
            });

            prop_assert!(res.is_ok(), "{:?}", res);
            let res = res.unwrap();
            prop_assert_eq!(1, res.len());
            let calc_steps = res[0].1;
            prop_assert_eq!(calc_steps, num_steps);
        }
    }
}

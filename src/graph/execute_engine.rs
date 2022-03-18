//! ## Engine lifecycle
//! 1. Create the `Engine` by passing it a `Graph`
//! 2. Call `run` or a variant to transform input representations to output representations
//! 3. GOTO 2
//!
//! Calling `run` requires a closure or an `OperationVisitor` trait object. This visitor will use
//! the operation data and input representation data to calculate output representations. It's
//! setup like this to facilitate the use of new representations.
use super::graph::{Graph, GraphState, OperationId, RepresentationId};
use super::operation::Operation;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;

pub struct Engine<'a> {
    graph: &'a Graph,
}

impl<'a> Engine<'a> {
    pub fn new(graph: &'a Graph) -> Self {
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
    pub fn run<T: Clone + Debug>(
        &self,
        output_ids: Vec<RepresentationId>,
        inputs: Vec<(RepresentationId, T)>,
        visit: impl FnMut(&dyn Operation, Vec<&T>, Option<usize>) -> (Option<usize>, Vec<T>),
    ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
        let mut state = ExecutionState::<T>::default();

        // Make sure that if visit_steps is false, then stepped inputs/outputs are not provided/asked for.
        // if !visit_steps {
        //     let stepped_inputs = inputs
        //         .iter()
        //         .filter(|(rep_id, _)| rep_id.operation_step.is_some())
        //         .collect::<Vec<_>>();
        //     let stepped_output_ids = output_ids
        //         .iter()
        //         .filter(|rep_id| rep_id.operation_step.is_some())
        //         .collect::<Vec<_>>();
        //     if !stepped_inputs.is_empty() || !stepped_output_ids.is_empty() {
        //         return Err(ExecuteError::VisitStepsTrueForSteppedInputsOrOutputs);
        //     }
        // }

        // Calculate subgraph and path of operations to perform
        // 1. Walk back through operations BFS
        let operation_set: HashSet<OperationId> = {
            // Set of representations that still need to be calculated
            let mut active_representation_ids = output_ids.clone();

            // set of ops indices required to produce the outputs
            let mut op_node_set: HashSet<OperationId> = HashSet::new();

            // Set of representations that have already been calculated by some previous op
            let mut finished_representations: HashSet<RepresentationId> =
                HashSet::from_iter(inputs.iter().map(|&(id, _)| id));

            while !active_representation_ids.is_empty() {
                // Get next representation we need
                let active_repr_id = active_representation_ids
                    .pop()
                    .ok_or(ExecuteError::PoppedEmptyStack)?;

                // The active id may already be added to finished representations if the producing
                // operation has more than one output.
                if finished_representations.contains(&active_repr_id) {
                    continue;
                }

                // If not, try to get it from an operation
                // `op_id` is the id of the operation that produces `active_repr_id`
                let op_id = self.graph.get_representation_op_id(&active_repr_id).ok_or(
                    ExecuteError::NoOpCreatesRepresentation {
                        repr_id: active_repr_id,
                    },
                )?;

                let op_node = self
                    .graph
                    .get_operation_node(&op_id)
                    .ok_or(ExecuteError::OperationNotExist { op_id })?;
                let op = op_node.get_operation();

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
                    return Err(ExecuteError::AnotherOpProducesOutput { op_id });
                }

                // Handle op inputs
                op_node.get_input_ids().iter().for_each(|&input_id| {
                    if !finished_representations.contains(&input_id) {
                        active_representation_ids.push(input_id);
                    }
                });
            }
            op_node_set
        };

        // 2. Order set via the graph's topological ordering.
        let mut op_node_vec: Vec<OperationId> = operation_set.into_iter().map(|x| x).collect();

        // This ordering is actually just the ascending ordering of OperationIds
        op_node_vec.sort();

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
                .map(|out_id| state.get_step_start(*out_id))
                .collect::<Option<Vec<_>>>()
                .unwrap_or(
                    op_node
                        .get_input_ids()
                        .iter()
                        .map(|x| x)
                        .collect::<Vec<_>>(),
                );
            let mut step = input_ids.first().unwrap().operation_step;

            // Collect input references
            let mut reprs = input_ids
                .iter()
                .map(|&id| state.get_representation(*id))
                .collect::<Option<Vec<&T>>>()
                .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                    repr_ids: op_node.get_input_ids().clone(),
                })?;

            // Loop over steps in the operation until the final output is returned.
            loop {
                // Calculate outputs of (step) operation
                let (new_step, outputs) = visit(op_node.get_operation().as_ref(), reprs, step);

                if outputs.len() != op_node.get_output_ids().len() {
                    return Err(ExecuteError::IncorrectOutputsFromVisitor {
                        expected: outputs.len(),
                        given: op_node.get_output_ids().len(),
                    });
                }

                // Store outputs
                for (&repr_id, repr) in op_node.get_output_ids().iter().zip(outputs.into_iter()) {
                    let repr_id = repr_id.clone();
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
                repr_ids: output_ids,
            })?;

        Ok(outputs)
    }

    pub fn run_graph_state_to<T: Clone + Debug>(
        &self,
        state: GraphState<T>,
        repr_id: RepresentationId,
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
    pub fn run_node_visitor<T: Clone + Debug>(
        &self,
        output_ids: Vec<RepresentationId>,
        inputs: Vec<(RepresentationId, T)>,
        visitor: &mut dyn OperationVisitor<T>,
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
}

pub trait OperationVisitor<T: Clone> {
    /// Visits the operations in set topological order. Will always visit child operations before parent operations.
    ///
    /// # Arguments
    ///
    /// * `operation` - The operation being visited
    /// * `inputs` - Inputs to the operation
    fn visit(&mut self, operation: &Box<dyn Operation>, inputs: Vec<&T>) -> Vec<T>;
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
    pub fn new(
        representations: HashMap<RepresentationId, T>,
        step_starts: &[RepresentationId],
        step_ends: &[RepresentationId],
    ) -> Self {
        Self {
            representations,
            step_starts: HashMap::from_iter(
                step_starts
                    .iter()
                    .map(|&repr_id| (repr_id.representation_node_id, repr_id)),
            ),
            step_ends: HashMap::from_iter(step_ends.iter().map(|repr_id| {
                (
                    repr_id.representation_node_id,
                    repr_id.operation_step.unwrap(),
                )
            })),
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
        // A representation should only ever be set once
        if self.representations.contains_key(&representation_id) {
            // TODO: Specify error
            Err(ExecuteError::GenericError)
        } else {
            self.representations
                .insert(representation_id, representation);
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub enum StepState<T: Clone + Debug> {
    Repr(T),                     // A single representation
    StepRepr(HashMap<usize, T>), // The representations within an operation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine2;
    use crate::bounds::Bounds1;
    use crate::graph::graph::OperationNode;
    use crate::star::Star2;
    use crate::NNVFloat;
    use ndarray::Array1;
    use ndarray::Array2;
    use serde::{Deserialize, Serialize};
    use std::any::Any;
    use std::fmt;
    use std::fmt::Debug;

    #[derive(Default, Clone, Debug, Serialize, Deserialize)]
    struct DummyOperation {
        op_id: OperationId,
    }

    impl DummyOperation {
        pub fn new(op_id: OperationId) -> Self {
            Self { op_id }
        }

        pub fn get_op_id(&self) -> OperationId {
            self.op_id
        }
    }

    #[typetag::serde]
    impl Operation for DummyOperation {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn num_steps(&self) -> Option<usize> {
            None
        }

        fn forward1(&self, input: &Vec<Array1<NNVFloat>>) -> Vec<Array1<NNVFloat>> {
            todo!()
        }

        fn forward2(&self, input: &Vec<Array2<NNVFloat>>) -> Vec<Array2<NNVFloat>> {
            todo!()
        }

        fn apply_bounds(
            &self,
            bounds: &Vec<Bounds1>,
            lower_aff: &Vec<Affine2>,
            upper_aff: &Vec<Affine2>,
        ) -> Vec<(Bounds1, Affine2, Affine2)> {
            todo!()
        }

        fn forward_star(
            &self,
            star: &Star2,
            activation_idx: Option<usize>,
            input_bounds: Option<Vec<Bounds1>>,
            parent_bounds: Option<Vec<Bounds1>>,
        ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool) {
            todo!()
        }

        fn construct_starnodetype(
            &self,
            child_ids: &[usize],
            dim: Option<usize>,
        ) -> crate::star_node::StarNodeType {
            todo!()
        }

        fn input_shapes(&self) -> Vec<crate::tensorshape::TensorShape> {
            panic!()
        }

        fn output_shapes(&self) -> Vec<crate::tensorshape::TensorShape> {
            panic!()
        }

        fn apply_bounds_step(
            &self,
            _dim: usize,
            _bounds: &Vec<Bounds1>,
            _lower_aff: &Vec<Affine2>,
            _upper_aff: &Vec<Affine2>,
        ) -> Vec<(Bounds1, Affine2, Affine2)> {
            panic!();
        }

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

        fn get_activation_pattern(
            &self,
            _state: Vec<&Array2<NNVFloat>>,
        ) -> Option<Vec<Array2<bool>>> {
            // This should only be Some in an activation layer (e.g. ReLU)
            None
        }
    }

    impl fmt::Display for DummyOperation {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "DummyOp")
        }
    }

    #[derive(Default)]
    struct OrderVisitor {
        order: Vec<OperationId>,
    }

    impl OrderVisitor {
        pub fn get_order(&self) -> &Vec<OperationId> {
            &self.order
        }
    }

    impl OperationVisitor<usize> for OrderVisitor {
        fn visit(&mut self, operation: &Box<dyn Operation>, _inputs: Vec<&usize>) -> Vec<usize> {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            self.order.push(op.get_op_id());
            vec![0]
        }
    }

    #[test]
    /// Tests the following graph structure where letters indicate tensors and -> indicate operations:
    /// Repr: A B C D E F
    /// ID:   0 1 2 3 4 5
    /// A -> B: Operation 0
    /// B -> C: Operation 1
    /// C -> D: Operation 2
    /// B -> E: Operation 3
    /// E -> F: Operation 4
    fn test_y_structure_graph() {
        let mut graph = Graph::default();
        let repr_ids = (0..6)
            .map(|id| RepresentationId::new(id, None))
            .collect::<Vec<_>>();
        let op_a_b = OperationNode::new(
            Box::new(DummyOperation::new(0)),
            vec![repr_ids[0]],
            vec![repr_ids[1]],
        );
        let op_b_c = OperationNode::new(
            Box::new(DummyOperation::new(1)),
            vec![repr_ids[1]],
            vec![repr_ids[2]],
        );
        let op_c_d = OperationNode::new(
            Box::new(DummyOperation::new(2)),
            vec![repr_ids[2]],
            vec![repr_ids[3]],
        );
        let op_b_e = OperationNode::new(
            Box::new(DummyOperation::new(3)),
            vec![repr_ids[1]],
            vec![repr_ids[4]],
        );
        let op_e_f = OperationNode::new(
            Box::new(DummyOperation::new(4)),
            vec![repr_ids[4]],
            vec![repr_ids[5]],
        );

        let op_a_b_id_res = graph.add_operation_node(op_a_b);
        let op_b_c_id_res = graph.add_operation_node(op_b_c);
        let op_c_d_id_res = graph.add_operation_node(op_c_d);
        let op_b_e_id_res = graph.add_operation_node(op_b_e);
        let op_e_f_id_res = graph.add_operation_node(op_e_f);

        assert!(op_a_b_id_res.is_ok());
        assert!(op_b_c_id_res.is_ok());
        assert!(op_c_d_id_res.is_ok());
        assert!(op_b_e_id_res.is_ok());
        assert!(op_e_f_id_res.is_ok());

        let op_a_b_id = op_a_b_id_res.unwrap();
        let op_b_c_id = op_b_c_id_res.unwrap();
        let op_c_d_id = op_c_d_id_res.unwrap();
        let op_b_e_id = op_b_e_id_res.unwrap();
        let op_e_f_id = op_e_f_id_res.unwrap();

        assert_eq!(op_a_b_id, 0);
        assert_eq!(op_b_c_id, 1);
        assert_eq!(op_c_d_id, 2);
        assert_eq!(op_b_e_id, 3);
        assert_eq!(op_e_f_id, 4);

        let engine = Engine::new(&graph);

        {
            // Test 1: Tests the whole graph
            let mut visitor = OrderVisitor::default();
            let run_res = engine.run_node_visitor(
                vec![repr_ids[3], repr_ids[5]],
                vec![(repr_ids[0], 0 as usize)],
                &mut visitor,
            );
            assert!(run_res.is_ok(), "{:?}", run_res);
            assert_eq!(visitor.order, vec![0 as OperationId, 1, 2, 3, 4]);
        }

        {
            // Test 2: Tests the subgraph (A -> B, B -> C, B -> E)
            let mut visitor = OrderVisitor::default();
            let run_res = engine.run_node_visitor(
                vec![repr_ids[2], repr_ids[4]],
                vec![(repr_ids[0], 0 as usize)],
                &mut visitor,
            );
            assert!(run_res.is_ok(), "{:?}", run_res);
            assert_eq!(visitor.order, vec![0 as OperationId, 1, 3]);
        }

        {
            // Test 3: Tests the subgraph (B -> C, C -> D)
            let mut visitor = OrderVisitor::default();
            let run_res = engine.run_node_visitor(
                vec![repr_ids[3]],
                vec![(repr_ids[1], 0 as usize)],
                &mut visitor,
            );
            assert!(run_res.is_ok(), "{:?}", run_res);
            assert_eq!(visitor.order, vec![1 as OperationId, 2]);
        }
    }
}

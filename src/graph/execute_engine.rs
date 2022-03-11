use super::graph::{Graph, Operation, OperationId, RepresentationId};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;

#[derive(Debug)]
pub enum ExecuteError {
    GenericError,
    PoppedEmptyStack,
    IncorrectOutputsFromVisitor { expected: usize, given: usize },
    NoOpCreatesRepresentation { repr_id: RepresentationId },
    OperationAddedTwice { op_id: OperationId },
    OperationNotExist { op_id: OperationId },
    AnotherOpProducesOutput { op_id: OperationId },
    OneOfRepresentationsNotExist { repr_ids: Vec<RepresentationId> },
    ReprIdGivenByInputsAndOpState { repr_ids: HashSet<RepresentationId> },
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
    representations: HashMap<RepresentationId, T>,
}

impl<T: Clone> Default for ExecutionState<T> {
    fn default() -> Self {
        Self {
            representations: HashMap::new(),
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

pub struct Engine<'a> {
    graph: &'a Graph,
}

impl<'a> Engine<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph }
    }

    /// Calculates output after a given step of an operation
    ///
    /// If an operation is required for the calculation and an input representation is provided for a
    /// specific step of the operation then all inputs for the operation at that step must be supplied.
    /// For instance, suppose we are calculating (A, B) -> C, i.e. C is calculated from an operation
    /// involving both A and B. Further, suppose the operation contains 3 steps and A is provided after
    /// performing step 2. Then B must also be provided after performing step 2.
    ///
    /// # Arguments
    ///
    /// * `outputs` - Representations along with steps to calculate. Use None to signify the end of the step.
    /// * `inputs` - The representations of inputs of the calculation.
    /// * `step_inputs` - Intermediate representation at a step in calculating the tensor of the given id.
    /// * `visit` - The function to run at each step of each operation. Note the extra arguments to the visit function.
    ///             The first is the index of the step if there are steps in the operation. The second is the output of
    ///             the previous step operation (or None for the first step operation).
    pub fn run_step<T: Clone + Debug>(
        &self,
        output_ids: Vec<RepresentationId>,
        output_state_ids: Vec<(OperationId, usize)>,
        inputs: Vec<(RepresentationId, T)>,
        operation_states: Vec<(OperationId, usize, Vec<T>)>,
        visit: impl FnMut(
            &Box<dyn Operation>,
            Option<Vec<&T>>,
            Option<usize>,
            Option<&Vec<T>>,
        ) -> Vec<T>,
    ) -> Result<Vec<(RepresentationId, StepState<T>)>, ExecuteError> {
        let mut internal_representations = HashMap::<OperationId, HashMap<usize, Vec<T>>>::new();

        // Check that the inputs from operation_states do not collide with those from inputs
        let operation_state_input_ids = operation_states
            .iter()
            .map(|(op_id, _, _)| {
                let op = self.graph.get_operation_node(*op_id).unwrap();
                op.get_output_ids().clone().into_iter()
            })
            .flatten()
            .collect::<HashSet<RepresentationId>>();
        let input_ids = inputs
            .iter()
            .map(|(repr_id, _)| *repr_id)
            .collect::<HashSet<RepresentationId>>();
        let id_intersection = operation_state_input_ids
            .intersection(&input_ids)
            .map(|&x| x)
            .collect::<HashSet<_>>();
        if !id_intersection.is_empty() {
            return Err(ExecuteError::ReprIdGivenByInputsAndOpState {
                repr_ids: id_intersection,
            });
        }

        // 1. Manually run forward step input operations
        let op_state_inputs = operation_states
            .into_iter()
            .map(|(op_id, step, in_repr)| {
                let mut reprs = HashMap::<usize, Vec<T>>::new();
                reprs.insert(step, in_repr);
                let op_node = self.graph.get_operation_node(op_id).unwrap();
                let op = op_node.get_operation();

                for s in (step + 1)..(op.output_dims() as usize) {
                    let step_inputs = reprs.get(&(s - 1));
                    let step_outputs = visit(op, None, Some(s), step_inputs);
                    reprs.insert(s, step_outputs);
                }

                internal_representations.insert(op_id, reprs);

                reprs
                    .get(&(op.output_dims() - 1))
                    .unwrap()
                    .iter()
                    .zip(op_node.get_output_ids().iter())
                    .map(|(repr, repr_id)| (*repr_id, repr.clone()))
            })
            .flatten();

        // 2. Collect all inputs together to run the rest of the visitor
        let inputs = inputs
            .into_iter()
            .chain(op_state_inputs)
            .collect::<Vec<_>>();

        // 3. Collect all output representation ids necessary.
        let op_state_output_repr_ids = output_state_ids.iter().map(|(op_id, _)| -> OperationId {
            let op = self.graph.get_operation_node(*op_id).unwrap();
            op.get_output_ids().first().unwrap().clone()
        });
        let output_ids = output_ids
            .iter()
            .map(|&x| x)
            .chain(op_state_output_repr_ids)
            .collect::<Vec<_>>();

        // 4. Run the visitor function for the graph
        let step_visit = |op: &Box<dyn Operation>, inp| -> Vec<T> {
            if op.is_activation() {
                let mut reprs = HashMap::<usize, Vec<T>>::new();

                (0..(op.output_dims() as usize)).for_each(|dim| {
                    let step_inputs = reprs.get(&(dim - 1));
                    let step_outputs = visit(op, inp, Some(dim), step_inputs);
                    reprs.insert(dim, step_outputs);
                });

                reprs.get(&(op.output_dims() - 1)).unwrap().clone()
            } else {
                todo!()
            }
        };
        let outputs = self.run(output_ids, inputs, step_visit);

        todo!();
    }

    /// Calculates the output of a visitor given the inputs
    ///
    /// # Arguments
    ///
    /// * `outputs` - Set of representations to calculate
    /// * `inputs` - Set of starting inputs required to calculate outputs
    /// * `visitor` - Performs the intermediate calculations at each node
    pub fn run_visitor<T: Clone + Debug>(
        &self,
        output_ids: Vec<RepresentationId>,
        inputs: Vec<(RepresentationId, T)>,
        visitor: &mut dyn OperationVisitor<T>,
    ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
        self.run(output_ids, inputs, |op, inp| visitor.visit(op, inp))
    }

    pub fn run<T: Clone + Debug>(
        &self,
        output_ids: Vec<RepresentationId>,
        inputs: Vec<(RepresentationId, T)>,
        visit: impl FnMut(&Box<dyn Operation>, Vec<&T>) -> Vec<T>,
    ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
        // Calculate subgraph and path of operations to perform
        // 1. Walk back through operations BFS
        let operation_set = {
            // Set of representations that still need to be calculated
            let mut active_representation_ids = output_ids.clone();

            // set of ops indices required to produce the outputs
            let mut op_node_set: HashSet<OperationId> = HashSet::new();

            // Set of representations that have already been calculated by some previous op
            let mut finished_representations: HashSet<RepresentationId> =
                HashSet::from_iter(inputs.iter().map(|&(id, _)| id));

            while !active_representation_ids.is_empty() {
                // Get next representation we need
                let active_repr_id: usize = active_representation_ids
                    .pop()
                    .ok_or(ExecuteError::PoppedEmptyStack)?;

                // The active id may already be added to finished representations if the producing
                // operation has more than one output.
                if finished_representations.contains(&active_repr_id) {
                    continue;
                }

                // If not, try to get it from an operation
                let op_id = self.graph.get_representation_op_id(active_repr_id).ok_or(
                    ExecuteError::NoOpCreatesRepresentation {
                        repr_id: active_repr_id,
                    },
                )?;

                if !op_node_set.insert(op_id) {
                    return Err(ExecuteError::OperationAddedTwice { op_id });
                }

                let op = self
                    .graph
                    .get_operation_node(op_id)
                    .ok_or(ExecuteError::OperationNotExist { op_id })?;

                // Handle op outputs
                if op
                    .get_output_ids()
                    .iter()
                    .map(|x| finished_representations.insert(*x))
                    .any(|x| !x)
                {
                    return Err(ExecuteError::AnotherOpProducesOutput { op_id });
                }

                // Handle op inputs
                op.get_input_ids().iter().for_each(|&input_id| {
                    if !finished_representations.contains(&input_id) {
                        active_representation_ids.push(input_id);
                    }
                });
            }
            op_node_set
        };

        // 2. Order set via the graph's topological ordering.
        //    This ordering is actually just the ascending ordering of OperationIds
        let mut op_node_vec: Vec<OperationId> = operation_set.into_iter().map(|x| x).collect();
        op_node_vec.sort();

        let mut state = ExecutionState::<T>::default();
        if inputs
            .iter()
            .map(|(id, v)| state.set_representation(*id, v.clone()))
            .any(|x| x.is_err())
        {
            return Err(ExecuteError::GenericError);
        }

        // 3. Apply the visitor to every operation in order
        for op_id in op_node_vec {
            let op_node = self
                .graph
                .get_operation_node(op_id)
                .ok_or(ExecuteError::OperationNotExist { op_id })?;

            // Collect input references
            let inputs = op_node
                .get_input_ids()
                .into_iter()
                .map(|&id| state.get_representation(id))
                .collect::<Option<Vec<&T>>>()
                .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                    repr_ids: op_node.get_input_ids().clone(),
                })?;

            // Execute visitor
            let outputs = visit(op_node.get_operation(), inputs);
            if outputs.len() != op_node.get_output_ids().len() {
                return Err(ExecuteError::IncorrectOutputsFromVisitor {
                    expected: outputs.len(),
                    given: op_node.get_output_ids().len(),
                });
            }

            // Store outputs
            for (&repr_id, repr) in op_node.get_output_ids().iter().zip(outputs.into_iter()) {
                state.set_representation(repr_id, repr)?;
            }
        }

        // Collect and return output representations
        let output_representations = output_ids
            .iter()
            .map(|&id| state.get_representation(id).cloned().map(|r| (id, r)))
            .collect::<Option<Vec<(RepresentationId, T)>>>()
            .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                repr_ids: output_ids,
            })?;

        Ok(output_representations)
    }
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

        fn forward1(&self, input: &Array1<NNVFloat>) -> Array1<NNVFloat> {
            todo!()
        }

        fn forward2(&self, input: &Array2<NNVFloat>) -> Array2<NNVFloat> {
            todo!()
        }

        fn apply_bounds(
            &self,
            bounds: &Bounds1,
            lower_aff: &Affine2,
            upper_aff: &Affine2,
        ) -> (Bounds1, (Affine2, Affine2)) {
            todo!()
        }

        fn forward_star(
            &self,
            star: &Star2,
            activation_idx: Option<usize>,
            input_bounds: Option<Bounds1>,
            parent_bounds: Option<Bounds1>,
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
        let op_a_b = OperationNode::new(
            Box::new(DummyOperation::new(0)),
            vec![0 as usize],
            vec![1 as usize],
        );
        let op_b_c = OperationNode::new(
            Box::new(DummyOperation::new(1)),
            vec![1 as usize],
            vec![2 as usize],
        );
        let op_c_d = OperationNode::new(
            Box::new(DummyOperation::new(2)),
            vec![2 as usize],
            vec![3 as usize],
        );
        let op_b_e = OperationNode::new(
            Box::new(DummyOperation::new(3)),
            vec![1 as usize],
            vec![4 as usize],
        );
        let op_e_f = OperationNode::new(
            Box::new(DummyOperation::new(4)),
            vec![4 as usize],
            vec![5 as usize],
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
            let run_res = engine.run(
                vec![3 as RepresentationId, 5 as RepresentationId],
                vec![(0 as RepresentationId, 0 as usize)],
                &mut visitor,
            );
            assert!(run_res.is_ok(), "{:?}", run_res);
            assert_eq!(visitor.order, vec![0 as OperationId, 1, 2, 3, 4]);
        }

        {
            // Test 2: Tests the subgraph (A -> B, B -> C, B -> E)
            let mut visitor = OrderVisitor::default();
            let run_res = engine.run(
                vec![2 as RepresentationId, 4 as RepresentationId],
                vec![(0 as RepresentationId, 0 as usize)],
                &mut visitor,
            );
            assert!(run_res.is_ok(), "{:?}", run_res);
            assert_eq!(visitor.order, vec![0 as OperationId, 1, 3]);
        }

        {
            // Test 3: Tests the subgraph (B -> C, C -> D)
            let mut visitor = OrderVisitor::default();
            let run_res = engine.run(
                vec![3 as RepresentationId],
                vec![(1 as RepresentationId, 0 as usize)],
                &mut visitor,
            );
            assert!(run_res.is_ok(), "{:?}", run_res);
            assert_eq!(visitor.order, vec![1 as OperationId, 2]);
        }
    }
}

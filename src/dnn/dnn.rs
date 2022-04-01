use crate::graph::{
    Engine, ExecuteError, Graph, GraphError, Operation, OperationId, RepresentationId,
};
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::Array2;
use ndarray::{Array1, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct DNN {
    graph: Graph,
    input_representation_ids: Vec<RepresentationId>,
    output_representation_ids: Vec<RepresentationId>,
}

/*
impl Clone for DNN {
    fn clone(&self) -> Self {
        DNN {
            layers: self
                .layers
                .iter()
                .map(|x| dyn_clone::clone_box(&*x))
                .collect(),
        }
    }
}
*/

impl DNN {
    pub fn new(
        graph: Graph,
        input_representation_ids: Vec<RepresentationId>,
        output_representation_ids: Vec<RepresentationId>,
    ) -> Self {
        Self {
            graph,
            input_representation_ids,
            output_representation_ids,
        }
    }

    /// # Panics
    /// TODO
    pub fn from_sequential(layers: &[Box<dyn Operation>]) -> Self {
        let mut graph = Graph::default();
        for (i, layer) in layers.iter().enumerate() {
            graph
                .add_operation(
                    layer.clone(),
                    vec![RepresentationId::new(i, None)],
                    vec![RepresentationId::new(i + 1, None)],
                )
                .unwrap();
        }
        let input_representation_ids = vec![RepresentationId::new(0, None)];
        let output_representation_ids = vec![RepresentationId::new(layers.len(), None)];
        Self {
            graph,
            input_representation_ids,
            output_representation_ids,
        }
    }

    /// # Errors
    pub fn add_operation(
        &mut self,
        op: Box<dyn Operation>,
        inputs: Vec<RepresentationId>,
        outputs: Vec<RepresentationId>,
    ) -> Result<OperationId, GraphError> {
        self.graph.add_operation(op, inputs, outputs)
    }

    pub fn get_operation(&self, id: OperationId) -> Option<&dyn Operation> {
        self.graph
            .get_operation_node(&id)
            .map(crate::graph::OperationNode::get_operation)
    }

    pub const fn get_input_representation_ids(&self) -> &Vec<RepresentationId> {
        &self.input_representation_ids
    }

    pub const fn get_output_representation_ids(&self) -> &Vec<RepresentationId> {
        &self.output_representation_ids
    }

    pub const fn get_graph(&self) -> &Graph {
        &self.graph
    }

    /// # Returns
    /// `Vec<(num_samples, dimension)>` where each entry is a layer's activations
    ///
    /// # Results
    /// TODO
    ///
    /// # Panics
    /// TODO
    pub fn calculate_activation_pattern2(
        &self,
        inputs: &[Array2<NNVFloat>],
    ) -> Result<Vec<Array2<bool>>, ExecuteError> {
        assert_eq!(inputs.len(), self.get_input_representation_ids().len());
        let mut activation_patterns = vec![];
        let inputs = self
            .get_input_representation_ids()
            .iter()
            .zip(inputs.iter())
            .map(|(&id, input)| (id, input.clone()))
            .collect::<Vec<_>>();
        Engine::new(&self.graph).run(
            self.get_output_representation_ids().clone(),
            &inputs,
            |op, inputs, _| -> (Option<usize>, Vec<Array2<NNVFloat>>) {
                if let Some(mut pattern) = op.get_activation_pattern(inputs) {
                    activation_patterns.append(&mut pattern);
                }
                (None, op.forward2(inputs))
            },
        )?;
        Ok(activation_patterns)
    }

    pub fn calculate_activation_pattern1(
        &self,
        inputs: &[Array1<NNVFloat>],
    ) -> Result<Vec<Array1<bool>>, ExecuteError> {
        Ok(self
            .calculate_activation_pattern2(
                &inputs
                    .iter()
                    .map(|input| input.clone().insert_axis(Axis(1)))
                    .collect::<Vec<_>>(),
            )?
            .into_iter()
            .map(|x| x.index_axis(Axis(1), 0).to_owned())
            .collect())
    }

    /// # Panics
    /// TODO
    pub fn forward1(&self, inputs: &[Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        assert_eq!(inputs.len(), self.get_input_representation_ids().len());
        let engine = Engine::new(&self.graph);
        let inputs = self
            .get_input_representation_ids()
            .iter()
            .zip(inputs.iter())
            .map(|(&id, input)| (id, input.clone()))
            .collect::<Vec<_>>();
        let res = engine.run(
            self.get_output_representation_ids().clone(),
            &inputs,
            |op: &dyn Operation, inputs, _| -> (Option<usize>, Vec<Array1<NNVFloat>>) {
                (None, op.forward1(inputs))
            },
        );
        res.unwrap().into_iter().map(|(_, output)| output).collect()
    }

    /// # Panics
    /// TODO
    pub fn forward_suffix1(
        &self,
        inputs: &[(RepresentationId, Array1<NNVFloat>)],
    ) -> Vec<Array1<NNVFloat>> {
        let engine = Engine::new(&self.graph);
        let res = engine.run(
            self.get_output_representation_ids().clone(),
            &inputs,
            |op: &dyn Operation, inputs, _| -> (Option<usize>, Vec<Array1<NNVFloat>>) {
                (None, op.forward1(inputs))
            },
        );
        res.unwrap().into_iter().map(|(_, output)| output).collect()
    }
}

impl DNN {
    /// # Panics
    /// TODO
    pub fn input_shapes(&self) -> Vec<TensorShape> {
        self.input_representation_ids
            .iter()
            .map(|repr_id| {
                let op_id = *self
                    .graph
                    .get_representation_input_op_ids(repr_id)
                    .first()
                    .unwrap();
                let op_node = self.graph.get_operation_node(&op_id).unwrap();
                let idx = op_node
                    .get_input_ids()
                    .iter()
                    .position(|x| *x == *repr_id)
                    .unwrap();
                op_node.get_operation().input_shapes()[idx].clone()
            })
            .collect::<Vec<_>>()
    }

    /// # Panics
    pub fn output_shapes(&self) -> Vec<TensorShape> {
        self.output_representation_ids
            .iter()
            .map(|repr_id| {
                let op_id = self.graph.get_representation_op_id(repr_id).unwrap();
                let op_node = self.graph.get_operation_node(&op_id).unwrap();
                let idx = op_node
                    .get_input_ids()
                    .iter()
                    .position(|x| *x == *repr_id)
                    .unwrap();
                op_node.get_operation().input_shapes()[idx].clone()
            })
            .collect::<Vec<_>>()

        // self.layers.last().unwrap().output_shape()
    }
}

impl fmt::Display for DNN {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // let layers: Vec<String> = self.layers.iter().map(|x| format!("{}", x)).collect();
        write!(
            f,
            "Input {:?} => {:?}",
            self.input_shapes(),
            // layers.join(" => ")
            self.output_shapes(),
        )
    }
}

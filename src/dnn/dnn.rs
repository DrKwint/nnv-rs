use crate::graph::{Graph, GraphError, Operation, OperationId, RepresentationId};
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
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

    pub fn from_sequential(layers: Vec<Box<dyn Operation>>) -> Self {
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
            .map(|node| node.get_operation().as_ref())
    }

    pub fn get_input_representation_ids(&self) -> &Vec<RepresentationId> {
        &self.input_representation_ids
    }

    pub fn get_output_representation_ids(&self) -> &Vec<RepresentationId> {
        &self.output_representation_ids
    }

    pub fn get_graph(&self) -> &Graph {
        &self.graph
    }

    // pub fn get_layer(&self, idx: usize) -> Option<&dyn Layer> {
    //     self.layers.get(idx).map(Box::as_ref)
    // }

    // pub fn get_layers(&self) -> &[Box<dyn Layer>] {
    //     &self.layers
    // }

    // pub fn iter(&self) -> DNNIterator {
    //     DNNIterator::new(self, DNNIndex::default())
    // }

    /// # Returns
    /// `Vec<(num_samples, dimension)>` where each entry is a layer's activations
    pub fn calculate_activation_pattern2(
        &self,
        _inputs: Vec<Array2<NNVFloat>>,
    ) -> Vec<Array2<bool>> {
        todo!();
        // self.layers
        //     .iter()
        //     .scan(input, |state, layer| {
        //         *state = layer.forward2(state);
        //         Some(layer.get_activation_pattern(state))
        //     })
        //     .flatten()
        //     .collect()
    }

    pub fn calculate_activation_pattern1(&self, _input: &Array1<NNVFloat>) -> Vec<Array1<bool>> {
        todo!();
        // self.calculate_activation_pattern2(input.clone().insert_axis(Axis(1)))
        //     .into_iter()
        //     .map(|x| x.index_axis(Axis(1), 0).to_owned())
        //     .collect()
    }

    pub fn forward1(&self, _input: Array1<NNVFloat>) -> Array1<NNVFloat> {
        todo!();
        // self.layers
        //     .iter()
        //     .fold(input, |x, layer| layer.forward1(&x))
    }

    pub fn forward_suffix1(
        &self,
        _input: Array1<NNVFloat>,
        // position: &DNNIndex,
    ) -> Array1<NNVFloat> {
        todo!();
        // self.layers
        //     .iter()
        //     .skip(position.get_layer_idx())
        //     .fold(input, |x, layer| layer.forward1(&x))
    }
}

impl DNN {
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
                    .into_iter()
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
                    .into_iter()
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

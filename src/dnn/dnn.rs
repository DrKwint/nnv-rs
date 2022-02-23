use super::dnn_iter::{DNNIndex, DNNIterator};
use super::layer::Layer;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct DNN {
    layers: Vec<Box<dyn Layer>>,
}

impl DNN {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn get_layer(&self, idx: usize) -> Option<&Box<dyn Layer>> {
        self.layers.get(idx)
    }

    pub fn get_layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn iter(&self) -> DNNIterator {
        DNNIterator::new(self, DNNIndex::default())
    }

    /// # Returns
    /// `Vec<(num_samples, dimension)>` where each entry is a layer's activations
    pub fn calculate_activation_pattern2(&self, input: Array2<NNVFloat>) -> Vec<Array2<bool>> {
        self.layers
            .iter()
            .scan(input, |state, layer| {
                *state = layer.forward2(state);
                Some(layer.get_activation_pattern(state))
            })
            .flatten()
            .collect()
    }

    pub fn calculate_activation_pattern1(&self, input: &Array1<NNVFloat>) -> Vec<Array1<bool>> {
        self.calculate_activation_pattern2(input.clone().insert_axis(Axis(1)))
            .into_iter()
            .map(|x| x.index_axis(Axis(1), 0).to_owned())
            .collect()
    }

    pub fn forward1(&self, input: Array1<NNVFloat>) -> Array1<NNVFloat> {
        self.layers
            .iter()
            .fold(input, |x, layer| layer.forward1(&x))
    }

    pub fn forward_suffix1(
        &self,
        input: Array1<NNVFloat>,
        position: &DNNIndex,
    ) -> Array1<NNVFloat> {
        self.layers
            .iter()
            .skip(position.get_layer_idx())
            .fold(input, |x, layer| layer.forward1(&x))
    }
}

impl DNN {
    pub fn input_shape(&self) -> TensorShape {
        self.layers[0].input_shape()
    }

    /// # Panics
    pub fn output_shape(&self) -> TensorShape {
        self.layers.last().unwrap().output_shape()
    }
}

impl fmt::Display for DNN {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layers: Vec<String> = self.layers.iter().map(|x| format!("{}", x)).collect();
        write!(f, "Input {} => {}", self.input_shape(), layers.join(" => "))
    }
}

use crate::affine::Affine;
use crate::affine::Affine2;
use crate::affine::Affine4;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly_relu;
use crate::star::Star;
use crate::star_node::StarNodeOp;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use log::trace;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Ix2;
use ndarray::Ix4;
use num::Zero;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct DNN {
    layers: Vec<Layer>,
}

impl DNN {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn get_layer(&self, idx: usize) -> Option<&Layer> {
        self.layers.get(idx)
    }

    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    /// # Returns
    /// `Vec<(num_samples, dimension)>` where each entry is a layer's activations
    pub fn calculate_activation_pattern(&self, input: Array2<NNVFloat>) -> Vec<Array2<bool>> {
        self.layers
            .iter()
            .scan(input, |state, layer| {
                *state = layer.forward2(state);
                let pattern_opt = match layer {
                    Layer::ReLU(_) => Some(state.mapv(|x| x >= 0.0)),
                    _ => None,
                };
                Some(pattern_opt)
            })
            .flatten()
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
            .skip(position.get_layer())
            .fold(input, |x, layer| layer.forward1(&x))
    }
}

impl DNN {
    pub fn input_shape(&self) -> TensorShape {
        self.layers[0].input_shape()
    }
}

impl fmt::Display for DNN {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layers: Vec<String> = self.layers.iter().map(|x| format!("{}", x)).collect();
        write!(f, "Input {} => {}", self.input_shape(), layers.join(" => "))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum Layer {
    Dense(Affine<Ix2>),
    Conv(Affine<Ix4>),
    BatchNorm(Affine<Ix2>),
    MaxPooling2D(usize),
    Flatten,
    ReLU(usize),
    Dropout(NNVFloat),
}

impl Layer {
    pub const fn new_dense(aff: Affine2) -> Self {
        Self::Dense(aff)
    }

    pub const fn new_conv(aff: Affine4) -> Self {
        Self::Conv(aff)
    }

    pub const fn new_maxpool(pool_size: usize) -> Self {
        Self::MaxPooling2D(pool_size)
    }

    pub const fn new_relu(num_dims: usize) -> Self {
        Self::ReLU(num_dims)
    }
}

impl Layer {
    /// # Panics
    pub fn input_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(aff) => TensorShape::new(vec![Some(aff.input_dim())]),
            Layer::Conv(conv_aff) => conv_aff.input_shape(),
            _ => panic!(),
        }
    }

    /// Give output shape as generically as possible
    ///
    /// # Panics
    pub fn output_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(aff) => TensorShape::new(vec![Some(aff.output_dim())]),
            _ => panic!(),
        }
    }

    /// Give the concrete output shape that results from giving an input with the specified shape
    ///
    /// # Panics
    // pub fn calculate_output_shape(&self, _input_shape: &TensorShape) -> TensorShape {

    pub fn forward1(&self, input: &Array1<NNVFloat>) -> Array1<NNVFloat> {
        match self {
            Layer::Dense(aff) => {
                debug_assert_eq!(input.ndim(), 1);
                aff.apply(&input.view())
            }
            Layer::ReLU(_) => input.mapv(|x| {
                if x.lt(&NNVFloat::zero()) {
                    NNVFloat::zero()
                } else {
                    x
                }
            }),
            _ => panic!(),
        }
    }

    /// # Panics
    pub fn forward2(&self, input: &Array2<NNVFloat>) -> Array2<NNVFloat> {
        match self {
            Layer::Dense(aff) => aff.apply_matrix(&input.view()),
            Layer::ReLU(_) => input.mapv(|x| {
                if x.lt(&NNVFloat::zero()) {
                    NNVFloat::zero()
                } else {
                    x
                }
            }),
            _ => panic!(),
        }
    }

    /// # Panics
    pub fn apply_star2(&self, star: &Star<Ix2>) -> Star<Ix2> {
        match self {
            Layer::Dense(aff) => star.affine_map2(aff),
            _ => panic!(),
        }
    }
}

impl Layer {
    /// # Panics
    pub fn apply_bounds(
        &self,
        lower_aff: &Affine2,
        upper_aff: &Affine2,
        bounds: &Bounds1,
    ) -> (Bounds1, (Affine2, Affine2)) {
        match self {
            Layer::Dense(aff) => {
                let new_lower = aff.signed_compose(lower_aff, upper_aff);
                let new_upper = aff.signed_compose(upper_aff, lower_aff);
                (aff.signed_apply(bounds), (new_lower, new_upper))
            }
            Layer::ReLU(ndims) => {
                if (ndims + 1) == bounds.ndim() {
                    deep_poly_relu(bounds, lower_aff, upper_aff)
                } else {
                    let (bounds_head, bounds_tail) = bounds.split_at(*ndims);
                    let (lower_aff_head, lower_aff_tail) = lower_aff.split_at(*ndims);
                    let (upper_aff_head, upper_aff_tail) = lower_aff.split_at(*ndims);
                    let (bounds_part, (lower_part, upper_part)) =
                        deep_poly_relu(&bounds_head, &lower_aff_head, &upper_aff_head);
                    (
                        bounds_part.append(&bounds_tail),
                        (
                            lower_part.append(&lower_aff_tail),
                            upper_part.append(&upper_aff_tail),
                        ),
                    )
                }
            }
            _ => panic!(),
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Layer::Dense(aff) => write!(f, "Dense {}", aff.output_dim()),
            Layer::Conv(conv_aff) => write!(f, "Conv {}", conv_aff.output_channels()),
            Layer::MaxPooling2D(pool_size) => write!(f, "MaxPool {}", pool_size),
            Layer::Flatten => write!(f, "Flatten"),
            Layer::ReLU(_) => write!(f, "ReLU"),
            _ => write!(f, "Missing Display for Layer!"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
/// Indicated which operations have already been run
pub struct DNNIndex {
    layer: Option<usize>,
    remaining_steps: Option<usize>,
}

impl DNNIndex {
    fn get_layer(&self) -> usize {
        self.layer.map_or(0, |x| x + 1)
    }

    fn increment(&mut self, dnn: &DNN) {
        // Decrement active relu
        let mut advance_layer_flag = false;
        if let Some(ref mut step) = self.remaining_steps {
            match step {
                0 => advance_layer_flag = true,
                x => *x -= 1,
            }
        } else {
            advance_layer_flag = true;
        }

        // advance layer at the end of running a full layer (e.g., all step relus)
        if advance_layer_flag {
            if let Some(ref mut layer) = self.layer {
                *layer += 1;
            } else {
                self.layer = Some(0);
            }

            if let Some(Layer::ReLU(ndims)) = dnn.get_layer(self.layer.unwrap()) {
                self.remaining_steps = Some(*ndims - 1);
            } else {
                self.remaining_steps = None;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DNNIterator<'a> {
    dnn: &'a DNN,
    idx: DNNIndex,
    finished: bool,
}

impl<'a> fmt::Display for DNNIterator<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DNNIterator(finished: {}, idx_layer: {:?}, idx_remaining_steps {:?})",
            self.finished, self.idx.layer, self.idx.remaining_steps
        )
    }
}

impl<'a> DNNIterator<'a> {
    pub const fn new(dnn: &'a DNN, idx: DNNIndex) -> Self {
        Self {
            dnn,
            idx,
            finished: false,
        }
    }

    pub const fn get_idx(&self) -> DNNIndex {
        self.idx
    }
}

impl Iterator for DNNIterator<'_> {
    type Item = StarNodeOp;

    fn next(&mut self) -> Option<Self::Item> {
        trace!("dnn iterator idx {:?}", self.get_idx());
        // Check finished
        if self.finished {
            return None;
        }

        // Increment index
        self.idx.increment(self.dnn);

        let layer_idx = self.idx.layer.as_mut().unwrap();

        // handle dropout after relu case
        if *layer_idx > 0 {
            if let (Some(Layer::ReLU(_)), Some(Layer::Dropout(_))) = (
                self.dnn.get_layer(*layer_idx - 1),
                self.dnn.get_layer(*layer_idx),
            ) {
                *layer_idx += 1;
            }
        }

        // Return operation
        if let Some(ref step) = self.idx.remaining_steps {
            if let Some(Layer::Dropout(prob)) = self.dnn.get_layer(*layer_idx + 1) {
                Some(StarNodeOp::StepReluDropout((*prob, *step)))
            } else {
                Some(StarNodeOp::StepRelu(*step))
            }
        } else {
            let layer = self.dnn.get_layer(*layer_idx);
            match layer {
                None => {
                    self.finished = true;
                    Some(StarNodeOp::Leaf)
                }
                Some(Layer::Dense(aff)) => Some(StarNodeOp::Affine(aff.clone())),
                Some(Layer::ReLU(_)) => {
                    panic!();
                }
                _ => panic!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{array1, fc_dnn};
    use proptest::prelude::*;

    #[test]
    fn test_dnn_index_increment() {
        let mut dnn = DNN::default();
        dnn.add_layer(Layer::new_relu(4));

        let mut idx = DNNIndex {
            layer: None,
            remaining_steps: None,
        };

        for i in (0..4).rev() {
            idx.increment(&dnn);
            assert_eq!(idx.layer.unwrap(), 0);
            assert_eq!(idx.remaining_steps.unwrap(), i);
        }

        idx.increment(&dnn);
        debug_assert!(dnn.get_layer(idx.layer.unwrap()).is_none());
    }

    proptest! {
        #[test]
        fn test_dnn_iterator_is_finite(dnn in fc_dnn(2, 2, 1, 2)) {
            let expected_steps: usize = dnn.layers.iter().enumerate().map(|(i, layer)| {
                match layer {
                    Layer::ReLU(ndims) => ndims,
                    Layer::Dropout(_) => {
                        if let Some(Layer::ReLU(_)) = dnn.get_layer(i - 1) {
                            &0
                        } else {
                            &1
                        }
                    },
                    _ => &1,
                }
            }).sum();

            let iter = DNNIterator::new(&dnn, DNNIndex{layer: None, remaining_steps: None});
            assert_eq!(iter.count(), expected_steps + 1);
        }
    }
}

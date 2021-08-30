use crate::affine::Affine2;
use crate::affine::Affine4;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly_relu;
use crate::star::Star;
use crate::star_node::StarNodeOp;
use crate::star_node::StarNodeType;
use crate::tensorshape::TensorShape;
use crate::Affine;
use ndarray::Array;
use ndarray::ArrayD;
use ndarray::Ix1;
use ndarray::Ix2;
use ndarray::Ix4;
use ndarray::IxDyn;
use ndarray::Zip;
use num::Float;
use std::fmt;
use std::iter::Sum;

#[derive(Default, Clone, Debug)]
pub struct DNN<T: num::Float> {
    layers: Vec<Layer<T>>,
}

impl<T: Float> DNN<T> {
    pub fn new(layers: Vec<Layer<T>>) -> Self {
        Self { layers }
    }

    pub fn add_layer(&mut self, layer: Layer<T>) {
        self.layers.push(layer)
    }

    pub fn get_layer(&self, idx: usize) -> Option<&Layer<T>> {
        self.layers.get(idx)
    }

    pub fn get_layers(&self) -> &[Layer<T>] {
        &self.layers
    }
}
impl<T> DNN<T>
where
    T: 'static
        + Float
        + std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    pub fn forward(&self, input: ArrayD<T>) -> ArrayD<T> {
        self.layers.iter().fold(input, |x, layer| layer.forward(x))
    }
}

impl<T> DNN<T>
where
    T: 'static
        + Float
        + std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    pub fn input_shape(&self) -> TensorShape {
        self.layers[0].input_shape()
    }
}

impl<T: 'static + num::Float> fmt::Display for DNN<T>
where
    T: 'static
        + Float
        + std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layers: Vec<String> = self.layers.iter().map(|x| format!("{}", x)).collect();
        write!(f, "Input {} => {}", self.input_shape(), layers.join(" => "))
    }
}

#[derive(Clone, Debug)]
pub enum Layer<T: num::Float> {
    Dense(Affine<T, Ix2>),
    Conv(Affine<T, Ix4>),
    BatchNorm(Affine<T, Ix2>),
    MaxPooling2D(usize),
    Flatten,
    ReLU(usize),
}

impl<T: Float> Layer<T> {
    pub fn new_dense(aff: Affine2<T>) -> Self {
        Self::Dense(aff)
    }

    pub fn new_conv(aff: Affine4<T>) -> Self {
        Self::Conv(aff)
    }

    pub fn new_maxpool(pool_size: usize) -> Self {
        Self::MaxPooling2D(pool_size)
    }

    pub fn new_relu(num_dims: usize) -> Self {
        Self::ReLU(num_dims)
    }
}

impl<T> Layer<T>
where
    T: 'static
        + Float
        + std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    pub fn input_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(aff) => TensorShape::new(vec![Some(aff.input_dim())]),
            Layer::Conv(conv_aff) => conv_aff.input_shape(),
            _ => panic!(),
        }
    }

    /// Give output shape as generically as possible
    pub fn output_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(aff) => TensorShape::new(vec![Some(aff.output_dim())]),
            _ => panic!(),
        }
    }

    /// Give the concrete output shape that results from giving an input with the specified shape
    pub fn calculate_output_shape(&self, _input_shape: &TensorShape) -> TensorShape {
        todo!();
    }

    pub fn forward2(&self, input: Array<T, Ix2>) -> Array<T, Ix2> {
        todo!();
    }

    pub fn forward(&self, input: ArrayD<T>) -> ArrayD<T> {
        match self {
            Layer::Dense(aff) => {
                assert_eq!(input.ndim(), 1);
                aff.apply(&input.into_dimensionality::<Ix1>().unwrap().view())
                    .into_dyn()
            }
            Layer::ReLU(_) => input.mapv(|x| if x.lt(&T::zero()) { T::zero() } else { x }),
            _ => panic!(),
        }
    }

    pub fn apply_star2(&self, star: Star<T, Ix2>) -> Star<T, Ix2> {
        match self {
            Layer::Dense(aff) => star.affine_map2(aff),
            _ => panic!(),
        }
    }
}

impl<T> Layer<T>
where
    T: 'static
        + Float
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign
        + Default
        + Sum,
{
    pub fn apply_bounds(
        &self,
        lower_aff: &Affine2<T>,
        upper_aff: &Affine2<T>,
        bounds: &Bounds1<T>,
    ) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
        match self {
            Layer::Dense(aff) => {
                let new_lower = aff.signed_compose(lower_aff, upper_aff);
                let new_upper = aff.signed_compose(upper_aff, lower_aff);
                (bounds.clone(), (new_lower, new_upper))
            }
            Layer::ReLU(_ndims) => {
                if (_ndims + 1) == bounds.ndim() {
                    deep_poly_relu(bounds, lower_aff, upper_aff)
                } else {
                    let (bounds_head, bounds_tail) = bounds.split_at(*_ndims);
                    let (lower_aff_head, lower_aff_tail) = lower_aff.split_at(*_ndims);
                    let (upper_aff_head, upper_aff_tail) = lower_aff.split_at(*_ndims);
                    let (bounds_part, (lower_part, upper_part)) =
                        deep_poly_relu(&bounds_head, &lower_aff_head, &upper_aff_head);
                    (
                        bounds_part.append(bounds_tail),
                        (
                            lower_part.append(lower_aff_tail),
                            upper_part.append(upper_aff_tail),
                        ),
                    )
                }
            }
            _ => panic!(),
        }
    }
}

impl<T: 'static + num::Float> fmt::Display for Layer<T> {
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

#[derive(Debug, Clone, Copy, Default)]
pub struct DNNIndex {
    layer: usize,
    remaining_steps: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct DNNIterator<'a, T: num::Float> {
    dnn: &'a DNN<T>,
    idx: DNNIndex,
    finished: bool,
}

impl<'a, T: Float> DNNIterator<'a, T> {
    pub fn new(dnn: &'a DNN<T>, idx: DNNIndex) -> Self {
        Self {
            dnn,
            idx,
            finished: false,
        }
    }

    pub fn get_idx(&self) -> DNNIndex {
        self.idx
    }
}

impl<T: num::Float> Iterator for DNNIterator<'_, T> {
    type Item = StarNodeOp<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if let Some(ref mut step) = self.idx.remaining_steps {
            *step -= 1;
            Some(StarNodeOp::StepRelu(*step))
        } else {
            let layer = self.dnn.get_layer(self.idx.layer);
            match layer {
                None => {
                    self.finished = true;
                    Some(StarNodeOp::Leaf)
                }
                Some(Layer::Dense(aff)) => Some(StarNodeOp::Affine(aff.clone())),
                Some(Layer::ReLU(ndim)) => {
                    self.idx.remaining_steps = Some(*ndim);
                    Some(StarNodeOp::StepRelu(*ndim))
                }
                _ => todo!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use proptest::prelude::{prop_assert, proptest};

    proptest! {
        #[test]
        fn test_relu_forward(arr in array1(4)) {
            let relu = Layer::new_relu(4);
            let out = relu.forward(arr.into_dyn());
            prop_assert!(out.iter().all(|&x| x >= 0.))
        }
    }
}

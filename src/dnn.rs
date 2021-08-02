use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly_relu;
use crate::star::Star;
use crate::tensorshape::TensorShape;
use crate::Affine;
use ndarray::Ix4;
use ndarray::IxDyn;
use ndarray::{Array1, Array2, Array4, Ix2};
use num::Float;
use std::fmt;

#[derive(Default, Clone, Debug)]
pub struct DNN<T: num::Float> {
    layers: Vec<Layer<T>>,
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
    pub fn add_layer(&mut self, layer: Layer<T>) {
        self.layers.push(layer)
    }

    pub fn input_shape(&self) -> TensorShape {
        self.layers[0].input_shape()
    }

    pub fn get_layer(&self, idx: usize) -> Option<&Layer<T>> {
        self.layers.get(idx)
    }

    pub fn get_layers(&self) -> &[Layer<T>] {
        &self.layers
    }
}

impl<T: 'static + num::Float> fmt::Display for DNN<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layers: Vec<String> = self.layers.iter().map(|x| format!("{}", x)).collect();
        write!(f, "{}", layers.join(" => "))
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
        todo!()
    }

    pub fn new_dense(mul: Array2<T>, shift: Array1<T>) -> Self {
        Self::Dense(Affine::<T, Ix2>::new(mul, shift))
    }

    pub fn new_conv(filters: Array4<T>, shift: Array1<T>) -> Self {
        Self::Conv(Affine::<T, Ix4>::new(filters, shift))
    }

    pub fn new_maxpool(pool_size: usize) -> Self {
        Self::MaxPooling2D(pool_size)
    }

    pub fn new_relu(num_dims: usize) -> Self {
        Self::ReLU(num_dims)
    }

    pub fn apply_star(&self, star: Star<T, IxDyn>) -> Star<T, IxDyn> {
        match self {
            Layer::Dense(aff) => {
                assert_eq!(star.ndim(), 2);
                star.into_dimensionality::<Ix2>()
                    .unwrap()
                    .affine_map2(aff)
                    .into_dyn()
            }
            _ => panic!(),
        }
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
        + std::ops::MulAssign
        + Default,
    f64: std::convert::From<T>,
{
    pub fn apply_bounds(
        &self,
        lower_aff: &Affine2<T>,
        upper_aff: &Affine2<T>,
        bounds: &Bounds1<T>,
    ) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
        match self {
            Layer::Dense(aff) => (bounds.clone(), (aff * lower_aff, aff * upper_aff)),
            Layer::ReLU(ndims) => deep_poly_relu(bounds, lower_aff, upper_aff),
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
            _ => write!(f, "Missing Display for Layer!"),
        }
    }
}

use crate::constellation::StarNode;
use crate::tensorshape::TensorShape;
use crate::Affine;
use ndarray::Ix4;
use ndarray::{Array1, Array2, Array4, Ix2};
use std::fmt;

#[derive(Default, Clone, Debug)]
pub struct DNN<T: num::Float> {
	layers: Vec<Layer<T>>,
}

impl<T: 'static + num::Float> DNN<T> {
	pub fn add_layer(&mut self, layer: Layer<T>) {
		self.layers.push(layer)
	}

	pub fn input_shape(&self) -> TensorShape {
		self.layers[0].input_shape().into()
	}

	pub fn get_layer(&self, idx: usize) -> Option<&Layer<T>> {
		self.layers.get(idx)
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

impl<T: 'static + num::Float> Layer<T> {
	pub fn input_shape(&self) -> TensorShape {
		match self {
			Layer::Dense(aff) => TensorShape::new(vec![Some(aff.input_dim())]),
			Layer::Conv(conv_aff) => conv_aff.input_shape(),
			_ => panic!(),
		}
	}

	pub fn new_dense(mul: Array2<T>, shift: Array1<T>) -> Self {
		Layer::Dense(Affine::<T, Ix2>::new(mul, shift))
	}

	pub fn new_conv(filters: Array4<T>, shift: Array1<T>) -> Self {
		Layer::Conv(Affine::<T, Ix4>::new(filters, shift))
	}

	pub fn new_maxpool(pool_size: usize) -> Self {
		Layer::MaxPooling2D(pool_size)
	}

	pub fn new_relu(num_dims: usize) -> Self {
		Layer::ReLU(num_dims)
	}

	pub fn apply(&self, starnode: &StarNode<T>) -> Vec<StarNode<T>> {
		match self {
			Layer::Dense(aff) => todo!(),
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

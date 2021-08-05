#![allow(clippy::module_name_repetitions)]
use crate::affine::Affine2;
use crate::rand::distributions::Distribution;
use crate::rand::SeedableRng;
use ndarray::iter::Lanes;
use ndarray::iter::LanesMut;
use ndarray::Array2;
use ndarray::ArrayView;
use ndarray::ArrayViewMut;
use ndarray::Axis;
use ndarray::Ix2;
use ndarray::Zip;
use ndarray::{stack, Array, Dimension};
use num::Float;
use rand::distributions::Uniform;
use rand::rngs::StdRng;

pub type Bounds1<T> = Bounds<T, Ix2>;

#[derive(Clone, Debug)]
pub struct Bounds<T: Float, D: Dimension> {
	data: Array<T, D>,
}

impl<T: Float, D: Dimension + ndarray::RemoveAxis> Bounds<T, D> {
	pub fn new<S: Dimension + Dimension<Larger = D>>(
		lower: Array<T, S>,
		upper: Array<T, S>,
	) -> Bounds<T, D> {
		let data: Array<T, D> = stack(Axis(0), &[lower.view(), upper.view()]).unwrap();
		Self { data }
	}

	pub fn lower(&self) -> ArrayView<T, D::Smaller> {
		self.data.index_axis(Axis(0), 0)
	}

	pub fn lower_mut(&mut self) -> ArrayViewMut<T, D::Smaller> {
		self.data.index_axis_mut(Axis(0), 0)
	}

	pub fn upper(&self) -> ArrayView<T, D::Smaller> {
		self.data.index_axis(Axis(0), 1)
	}

	pub fn upper_mut(&mut self) -> ArrayViewMut<T, D::Smaller> {
		self.data.index_axis_mut(Axis(0), 1)
	}

	pub fn ndim(&self) -> usize {
		self.data.shape().iter().skip(1).product()
	}

	pub fn bounds_iter(&self) -> Lanes<T, D::Smaller> {
		self.data.lanes(Axis(0))
	}

	pub fn bounds_iter_mut(&mut self) -> LanesMut<T, D::Smaller> {
		self.data.lanes_mut(Axis(0))
	}

	pub fn is_member(&self, x: &ArrayView<T, D::Smaller>) -> bool {
		Zip::from(x)
			.and(self.bounds_iter())
			.all(|&x, bounds| bounds[0] < x && x < bounds[1])
	}
}

impl<
		T: Float + rand::distributions::uniform::SampleUniform,
		D: Dimension + ndarray::RemoveAxis,
	> Bounds<T, D>
{
	pub fn sample_uniform(&self, seed: u64) -> Array<T, D::Smaller> {
		let mut rng = StdRng::seed_from_u64(seed);
		Zip::from(self.bounds_iter())
			.map_collect(|x| Uniform::new_inclusive(x[0], x[1]).sample(&mut rng))
	}
}

impl<T: 'static + Float + Default> Bounds1<T> {
	pub fn default(dim: usize) -> Self {
		Self {
			data: Array2::default([2, dim]),
		}
	}

	pub fn affine_map(&self, aff: Affine2<T>) -> Self {
		let lower = aff.apply(&self.lower());
		let upper = aff.apply(&self.upper());
		Self::new(lower, upper)
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use crate::test_util::bounds1;
	use proptest::proptest;

	proptest! {
		#[test]
		fn ordering_after_affine(bounds in bounds1(4)) {
			bounds; // todo!
		}

	}
}

extern crate ndarray;
extern crate ndarray_rand;

pub mod affine;
pub mod constellation;
pub mod polytope;
pub mod star;
pub mod util;
use crate::affine::Affine;
use crate::constellation::Constellation;
use crate::star::Star;
use ndarray::concatenate;
use ndarray::Axis;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

fn main() {
	let ndim = 3;
	let dist = Normal::new(0., 1.).unwrap();
	let generate_layer = |in_, out_| {
		Affine::new(
			Array2::random((in_, out_), dist),
			Array1::random(out_, dist),
		)
	};
	let random_part = Array1::random(2, dist);
	let lbs = concatenate(
		Axis(0),
		&[random_part.view(), Array1::from_elem(1, -2.).view()],
	)
	.unwrap();
	let ubs = concatenate(
		Axis(0),
		&[random_part.view(), Array1::from_elem(1, 2.).view()],
	)
	.unwrap();
	let star = Star::default(ndim).with_input_bounds(lbs, ubs);
	let a = generate_layer(ndim, 5);
	let b = generate_layer(5, 2);
	let c = generate_layer(2, 1);
	let mut constellation = Constellation::new(star, vec![a, b, c]);

	let loc = Array1::zeros(ndim);
	let scale = Array2::eye(ndim);
	let val = constellation.sample(&loc, &scale, -10., 10000, 5);
	println!("{:?}", val);
}

use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::DNN;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::ScalarOperand;
use ndarray::Zip;
use num::Float;
use std::fmt::Debug;
use std::ops::MulAssign;

pub fn deep_poly_relu<
	T: Float + Default + std::ops::MulAssign + ScalarOperand + std::ops::Mul + std::ops::Add,
>(
	bounds: &Bounds1<T>,
	lower_aff: &Affine2<T>,
	upper_aff: &Affine2<T>,
) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
	let mut out = Bounds1::default(bounds.ndim());
	let mut l_mul = Array1::ones(bounds.ndim());
	let mut u_mul = Array1::ones(bounds.ndim());
	let mut u_shift = Array1::zeros(bounds.ndim());
	Zip::from(bounds.bounds_iter())
		.and(out.bounds_iter_mut())
		.and(&mut l_mul)
		.and(&mut u_mul)
		.and(&mut u_shift)
		.for_each(
			|b: ArrayView1<T>,
			 mut out: ArrayViewMut1<T>,
			 l_mul: &mut T,
			 u_mul: &mut T,
			 u_shift: &mut T| {
				let l = b[0];
				let u = b[1];
				if u < T::zero() {
					out[0] = T::zero();
					out[1] = T::zero();
					*l_mul = T::zero();
					*u_mul = T::zero();
				} else if l > T::zero() {
					// Leave mul and shift at defaults
					out[0] = l;
					out[1] = u;
				} else {
					out[1] = u;
					*u_mul = u / (u - l);
					*u_shift = T::neg((u * l) / (u - l));
					// use approximation with least area
					if u < T::neg(l) {
						// Eqn. 3 from the paper
						out[0] = T::zero();
						*l_mul = T::zero();
					} else {
						// Eqn. 4 from the paper, leave l_mul at default
						out[0] = l;
					}
				}
			},
		);
	let lower_aff = lower_aff.clone() * l_mul;
	let upper_aff = (upper_aff.clone() * u_mul) + u_shift;
	(out, (lower_aff, upper_aff))
}

pub fn deep_poly<T: 'static + Float>(input_bounds: Bounds1<T>, dnn: &DNN<T>) -> Bounds1<T>
where
	f64: std::convert::From<T>,
	T: std::convert::From<f64>
		+ std::convert::Into<f64>
		+ ndarray::ScalarOperand
		+ std::fmt::Display
		+ Debug
		+ MulAssign
		+ Default,
{
	let ndim = input_bounds.ndim();
	// Affine expressing bounds on each variable in current layer as a
	// linear function of input bounds
	let aff_bounds = dnn.get_layers().iter().fold(
		(Affine2::identity(ndim), Affine2::identity(ndim)),
		|(laff, uaff), layer| {
			// Substitute input concrete bounds into current abstract bounds
			// to get current concrete bounds
			let bounds_concrete = Bounds1::new(
				laff.apply(&input_bounds.lower()),
				uaff.apply(&input_bounds.upper()),
			);
			// Calculate new abstract bounds from concrete bounds and layer
			let out = layer.apply_bounds(&laff, &uaff, &bounds_concrete);
			out.1
		},
	);
	// Final substitution to get output bounds
	Bounds1::new(
		aff_bounds.0.apply(&input_bounds.lower()),
		aff_bounds.1.apply(&input_bounds.upper()),
	)
}

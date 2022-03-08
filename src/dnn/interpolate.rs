#![allow(non_snake_case, clippy::module_name_repetitions)]
//! Representation of affine transformations
use super::layer::Layer;
use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::star::Star2;
use crate::star_node::StarNodeType;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Debug;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum InterpolateMethod {
    Bilinear,
}

/// Assumes that data is always in a flattened state.
/// Weights are of the shape: (kernel_w, kernel_h, channels_in, channels_out)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Interpolate {
    input_shape: TensorShape,
    output_shape: TensorShape,
    method: InterpolateMethod,
    affine: Option<Affine2>,
}

impl Interpolate {
    /// # Panics
    /// If improper shapes are passed in
    pub fn new(
        input_shape: TensorShape,
        output_shape: TensorShape,
        method: InterpolateMethod,
    ) -> Self {
        assert_eq!(input_shape[3].unwrap(), output_shape[3].unwrap());
        let mut s = Self {
            input_shape,
            output_shape,
            method,
            affine: None,
        };
        s.construct_affine();
        s
    }

    pub fn get_affine(&self) -> &Affine2 {
        self.affine.as_ref().unwrap()
    }

    pub fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    pub fn output_shape(&self) -> TensorShape {
        self.output_shape.clone()
    }

    fn construct_affine(&mut self) {
        let h_in = self.input_shape[1].unwrap();
        let w_in = self.input_shape[2].unwrap();
        let c = self.input_shape[3].unwrap();
        let h_out = self.output_shape()[1].unwrap();
        let w_out = self.output_shape()[2].unwrap();
        debug_assert!(h_in > 0);
        debug_assert!(w_in > 0);
        debug_assert!(c > 0);
        debug_assert!(h_out > 0);
        debug_assert!(w_out > 0);

        let input_dims = h_in * w_in * c;
        let output_dims = h_out * w_out * c;

        let mut weight = Array2::<NNVFloat>::zeros((output_dims, input_dims));
        for ((y_out, x_out), c_out) in (0..h_out)
            .cartesian_product(0..w_out)
            .cartesian_product(0..c)
        {
            let mut x_1: usize = 0;
            let mut x_2: usize = 0;
            let mut y_1: usize = 0;
            let mut y_2: usize = 0;
            if w_out > 1 {
                x_1 = (x_out as f64 * (w_in - 1) as f64 / (w_out - 1) as f64).floor() as usize;
                x_2 = (x_out as f64 * (w_in - 1) as f64 / (w_out - 1) as f64).ceil() as usize;
            }
            if h_out > 1 {
                y_1 = (y_out as f64 * (h_in - 1) as f64 / (h_out - 1) as f64).floor() as usize;
                y_2 = (y_out as f64 * (h_in - 1) as f64 / (h_out - 1) as f64).ceil() as usize;
            }

            let output_idx = y_out * (w_out * c) + x_out * c + c_out;

            if x_1 == x_2 && y_1 == y_2 {
                let input_idx = y_1 * (w_in * c) + x_1 * c + c_out;
                weight[[output_idx, input_idx]] = 1.;
            } else if x_1 == x_2 {
                let input_idx_1 = y_1 * (w_in * c) + x_1 * c + c_out;
                let input_idx_2 = y_2 * (w_in * c) + x_2 * c + c_out;
                let prop_width = x_2 as f64 / w_out as f64 - x_1 as f64 / w_in as f64;
                let weight_1 =
                    (x_out as f64 / w_out as f64 - x_1 as f64 / w_in as f64) / prop_width;
                let weight_2 =
                    (x_2 as f64 / w_out as f64 - x_out as f64 / w_in as f64) / prop_width;
                weight[[output_idx, input_idx_1]] = weight_1;
                weight[[output_idx, input_idx_2]] = weight_2;
            } else if y_1 == y_2 {
                let input_idx_1 = y_1 * (w_in * c) + x_1 * c + c_out;
                let input_idx_2 = y_2 * (w_in * c) + x_2 * c + c_out;
                let prop_height = y_2 as f64 / h_out as f64 - y_1 as f64 / h_in as f64;
                let weight_1 =
                    (y_out as f64 / h_out as f64 - y_1 as f64 / h_in as f64) / prop_height;
                let weight_2 =
                    (y_2 as f64 / h_out as f64 - y_out as f64 / h_in as f64) / prop_height;
                weight[[output_idx, input_idx_1]] = weight_1;
                weight[[output_idx, input_idx_2]] = weight_2;
            } else {
                let input_idx_11 = y_1 * (w_in * c) + x_1 * c + c_out;
                let input_idx_12 = y_1 * (w_in * c) + x_2 * c + c_out;
                let input_idx_21 = y_2 * (w_in * c) + x_1 * c + c_out;
                let input_idx_22 = y_2 * (w_in * c) + x_2 * c + c_out;

                let prop_width = x_2 as f64 / w_out as f64 - x_1 as f64 / w_in as f64;
                let weight_x_1 =
                    (x_out as f64 / w_out as f64 - x_1 as f64 / w_in as f64) / prop_width;
                let weight_x_2 =
                    (x_2 as f64 / w_out as f64 - x_out as f64 / w_in as f64) / prop_width;

                let prop_height = y_2 as f64 / h_out as f64 - y_1 as f64 / h_in as f64;
                let weight_y_1 =
                    (y_out as f64 / h_out as f64 - y_1 as f64 / h_in as f64) / prop_height;
                let weight_y_2 =
                    (y_2 as f64 / h_out as f64 - y_out as f64 / h_in as f64) / prop_height;

                weight[[output_idx, input_idx_11]] = weight_y_1 * weight_x_1;
                weight[[output_idx, input_idx_12]] = weight_y_1 * weight_x_2;
                weight[[output_idx, input_idx_21]] = weight_y_2 * weight_x_1;
                weight[[output_idx, input_idx_22]] = weight_y_2 * weight_x_2;
            }
        }
        let bias = Array1::<NNVFloat>::zeros(h_out * w_out * c);
        self.affine = Some(Affine2::new(weight, bias))
    }
}

#[typetag::serde]
impl Layer for Interpolate {
    fn input_shape(&self) -> TensorShape {
        TensorShape::new(vec![Some(self.get_affine().input_dim())])
    }

    fn output_shape(&self) -> TensorShape {
        TensorShape::new(vec![Some(self.get_affine().output_dim())])
    }
    fn forward1(&self, input: &Array1<NNVFloat>) -> Array1<NNVFloat> {
        debug_assert_eq!(input.ndim(), 1);
        self.get_affine().apply(&input.view())
    }

    fn forward2(&self, input: &Array2<NNVFloat>) -> Array2<NNVFloat> {
        self.get_affine().apply_matrix(&input.view())
    }

    fn apply_bounds(
        &self,
        bounds: &Bounds1,
        lower_aff: &Affine2,
        upper_aff: &Affine2,
    ) -> (Bounds1, (Affine2, Affine2)) {
        let new_lower = self.get_affine().signed_compose(lower_aff, upper_aff);
        let new_upper = self.get_affine().signed_compose(upper_aff, lower_aff);
        (
            self.get_affine().signed_apply(bounds),
            (new_lower, new_upper),
        )
    }

    fn forward_star(
        &self,
        star: &Star2,
        _activation_idx: Option<usize>,
        _input_bounds: Option<Bounds1>,
        _parent_bounds: Option<Bounds1>,
    ) -> (Vec<Star2>, Vec<Option<Bounds1>>, bool) {
        (vec![star.affine_map2(self.get_affine())], vec![None], false)
    }

    fn construct_starnodetype(&self, child_ids: &[usize], _dim: Option<usize>) -> StarNodeType {
        debug_assert_eq!(child_ids.len(), 1);
        StarNodeType::Conv {
            child_idx: child_ids[0],
        }
    }
}

impl fmt::Display for Interpolate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Interpolate {} -> {} {:?}",
            self.input_shape, self.output_shape, self.method
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use ndarray::Array3;
    use proptest::prelude::*;
    use std::panic;

    prop_compose! {
        fn generic_3d_tensor(max_width: usize, max_height:usize, max_channels:usize)
            (width in 1..max_width, height in 1..max_height, channels in 1..max_channels)
            (data in array3(width, height, channels)) -> Array3<NNVFloat> {
            data
        }
    }

    proptest! {
        #[test]
        fn test_interpolate_sizes(data in generic_3d_tensor(11, 11, 11), h_out in 1..11usize, w_out in 1..11usize) {
            let h_in = data.shape()[0];
            let w_in = data.shape()[1];
            let c = data.shape()[2];

            let layer = Interpolate::new(TensorShape::new(vec![None, Some(h_in), Some(w_in), Some(c)]),
                TensorShape::new(vec![None, Some(h_out), Some(w_out), Some(c)]), InterpolateMethod::Bilinear);

            let input_dims = h_in * w_in * c;
            let flat_data = data.view().into_shape(input_dims).unwrap();
            let output_dims = h_out * w_out * c;

            let affine_res = panic::catch_unwind(|| {
                layer.get_affine()
            });
            prop_assert!(affine_res.is_ok());
            let affine = affine_res.unwrap();
            let affine_result = affine.apply(&flat_data);

            prop_assert_eq!(affine_result.shape().len(), 1);
            prop_assert_eq!(affine_result.shape()[0], output_dims);
        }
    }
}

#![allow(non_snake_case, clippy::module_name_repetitions)]
//! Representation of affine transformations
use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::graph::Operation;
use crate::star::Star2;
// use crate::star::Star2;
use crate::tensorshape::TensorShape;
use crate::NNVFloat;
use itertools::Itertools;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView3};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::ops::Deref;

/// Assumes that data is always in a flattened state.
/// Weights are of the shape: (`kernel_w`, `kernel_h`, `channels_in`, `channels_out`)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Conv {
    kernel: Array4<NNVFloat>, // (K_h, K_w, C_in, C_out) following tf convention
    bias: Array1<NNVFloat>,   // (C_out)
    input_shape: TensorShape, // (None, H, W, C_in)
    strides: (usize, usize, usize), // (3)
    padding: ((usize, usize), (usize, usize)), // ((top, bottom), (left, right))
    affine: Option<Affine2>,
}

impl Conv {
    /// # Panics
    /// If improper shapes are passed in
    pub fn new(
        kernel: Array4<NNVFloat>,
        bias: Array1<NNVFloat>,
        input_shape: TensorShape,
        strides: (usize, usize, usize),
        padding: ((usize, usize), (usize, usize)),
    ) -> Self {
        debug_assert_eq!(kernel.shape()[3], bias.len());
        if strides.2 != 1 {
            todo!();
        }
        let mut s = Self {
            kernel,
            bias,
            input_shape,
            strides,
            padding,
            affine: None,
        };
        s.construct_affine();
        s
    }

    /// # Panics
    pub fn get_affine(&self) -> &Affine2 {
        self.affine.as_ref().unwrap()
    }

    pub fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    /// # Panics
    pub fn output_shape(&self) -> TensorShape {
        let k_h = self.kernel.shape()[0];
        let k_w = self.kernel.shape()[1];
        let h_out =
            (self.input_shape[1].unwrap() + self.padding.0 .0 + self.padding.0 .1 - (k_h - 1) - 1)
                / self.strides.0
                + 1;
        let w_out =
            (self.input_shape[2].unwrap() + self.padding.1 .0 + self.padding.1 .1 - (k_w - 1) - 1)
                / self.strides.1
                + 1;

        TensorShape::new(vec![
            None,
            Some(h_out),
            Some(w_out),
            Some(self.kernel.shape()[3]),
        ])
    }

    /// # Panics
    fn construct_affine(&mut self) {
        let h_in = self.input_shape[1].unwrap();
        let w_in = self.input_shape[2].unwrap();
        let c_in = self.input_shape[3].unwrap();
        let h_out = self.output_shape()[1].unwrap();
        let w_out = self.output_shape()[2].unwrap();
        let c_out = self.output_shape()[3].unwrap();
        let k_h = self.kernel.shape()[0];
        let k_w = self.kernel.shape()[1];

        let input_dims = h_in * w_in * c_in;
        let output_dims = h_out * w_out * c_out;

        let mut weight = Array2::<NNVFloat>::zeros((output_dims, input_dims));
        for (y_out, x_out) in (0..h_out).cartesian_product(0..w_out) {
            let y_0 = y_out * self.strides.0;
            let x_0 = x_out * self.strides.1;

            // Assign each filter of a pixel in the output to have the kernel contents
            for k_y in 0..k_h {
                if y_0 + k_y < self.padding.0 .0 || y_0 + k_y >= h_in + self.padding.0 .0 {
                    // Assumption that padding value is 0, so continue;
                    continue;
                }
                let y_in = y_0 + k_y - self.padding.0 .0;
                for k_x in 0..k_w {
                    if x_0 + k_x < self.padding.1 .0 || x_0 + k_x >= w_in + self.padding.1 .0 {
                        // Assumption that padding value is 0, so continue;
                        continue;
                    }
                    let x_in = x_0 + k_x - self.padding.1 .0;

                    for f_in in 0..c_in {
                        let input_idx = y_in * (w_in * c_in) + x_in * c_in + f_in;

                        for f_out in 0..c_out {
                            let output_idx = y_out * (w_out * c_out) + x_out * c_out + f_out;
                            weight[[output_idx, input_idx]] = self.kernel[[k_y, k_x, f_in, f_out]];
                        }
                    }
                }
            }
        }

        let bias = (Array3::<NNVFloat>::ones((h_out, w_out, c_out))
            * self.bias.view().into_shape((1, 1, c_out)).unwrap())
        .into_shape(h_out * w_out * c_out)
        .unwrap();

        self.affine = Some(Affine2::new(weight, bias));
    }

    /// # Panics
    pub fn convolve(&self, data: ArrayView3<NNVFloat>) -> Array3<NNVFloat> {
        let h_in = self.input_shape[1].unwrap();
        let w_in = self.input_shape[2].unwrap();
        let c_in = self.input_shape[3].unwrap();
        let h_out = self.output_shape()[1].unwrap();
        let w_out = self.output_shape()[2].unwrap();
        let c_out = self.output_shape()[3].unwrap();
        let k_h = self.kernel.shape()[0];
        let k_w = self.kernel.shape()[1];

        let input_shape = vec![h_in, w_in, c_in];
        let output_shape = (h_out, w_out, c_out);

        assert_eq!(data.shape(), input_shape);
        let mut output = Array3::<NNVFloat>::ones(output_shape);
        output = output * self.bias.view().into_shape((1, 1, c_out)).unwrap();

        for (y_out, x_out) in (0..h_out).cartesian_product(0..w_out) {
            let y_0 = y_out * self.strides.0;
            let x_0 = x_out * self.strides.1;

            // Assign each filter of a pixel in the output to have the kernel contents
            for k_y in 0..k_h {
                if y_0 + k_y < self.padding.0 .0 || y_0 + k_y >= h_in + self.padding.0 .0 {
                    continue;
                }
                let y_in = y_0 + k_y - self.padding.0 .0;
                for k_x in 0..k_w {
                    if x_0 + k_x < self.padding.1 .0 || x_0 + k_x >= w_in + self.padding.1 .0 {
                        continue;
                    }
                    let x_in = x_0 + k_x - self.padding.1 .0;

                    for f_in in 0..c_in {
                        for f_out in 0..c_out {
                            output[[y_out, x_out, f_out]] +=
                                data[[y_in, x_in, f_in]] * self.kernel[[k_y, k_x, f_in, f_out]];
                        }
                    }
                }
            }
        }
        output
    }
}

impl Operation for Conv {
    fn input_shapes(&self) -> Vec<TensorShape> {
        vec![TensorShape::new(vec![Some(self.get_affine().input_dim())])]
    }

    fn output_shapes(&self) -> Vec<TensorShape> {
        vec![TensorShape::new(vec![Some(self.get_affine().output_dim())])]
    }

    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        assert_eq!(input.len(), 1);
        let input = input.first().unwrap();
        debug_assert_eq!(input.ndim(), 1);
        vec![self.get_affine().apply(&input.view())]
    }

    fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
        assert_eq!(input.len(), 1);
        let input = input.first().unwrap();
        vec![self.get_affine().apply_matrix(&input.view())]
    }

    fn apply_bounds(
        &self,
        bounds: &[&Bounds1],
        lower_aff: &[&Affine2],
        upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        assert_eq!(bounds.len(), 1);
        assert_eq!(lower_aff.len(), 1);
        assert_eq!(upper_aff.len(), 1);
        let bounds = bounds.first().unwrap();
        let lower_aff = lower_aff.first().unwrap();
        let upper_aff = upper_aff.first().unwrap();
        let new_lower = self.get_affine().signed_compose(lower_aff, upper_aff);
        let new_upper = self.get_affine().signed_compose(upper_aff, lower_aff);
        vec![(self.get_affine().signed_apply(bounds), new_lower, new_upper)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        stars: Vec<StarRef>,
        _activation_idx: Option<usize>,
        _input_bounds: &Bounds1,
        parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>> {
        assert_eq!(1, stars.len());
        assert!(parent_local_output_bounds_opt.map_or(true, |b| b.len() == 1));
        vec![vec![(stars[0].affine_map2(self.get_affine()), None)]]
    }
}

impl fmt::Display for Conv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Conv {}x{}, {}",
            self.kernel.shape()[1],
            self.kernel.shape()[0],
            self.kernel.shape()[2]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use proptest::prelude::*;

    #[must_use = "strategies do nothing unless used"]
    pub fn conv_test_inputs(
        max_k_w: usize,
        max_k_h: usize,
        max_w: usize,
        max_h: usize,
        max_c_in: usize,
        max_c_out: usize,
        max_stride_x: usize,
        max_stride_y: usize,
    ) -> impl Strategy<Value = (Conv, Array3<NNVFloat>)> {
        let strat = (
            1..(max_k_w + 1),
            1..(max_k_h + 1),
            1..(max_c_in + 1),
            1..(max_c_out + 1),
        );
        let strat = Strategy::prop_flat_map(strat, move |(k_w, k_h, c_in, c_out)| {
            (
                Just(k_w),
                Just(k_h),
                Just(c_in),
                Just(c_out),
                k_w..(max_w + 1),
                k_h..(max_h + 1),
            )
        });
        let strat = Strategy::prop_flat_map(strat, move |(k_w, k_h, c_in, c_out, w_in, h_in)| {
            (
                array4(k_h, k_w, c_in, c_out),
                array3(h_in, w_in, c_in),
                array1(c_out),
                0..(k_h / 2 + 1),
                0..(k_w / 2 + 1),
                0..(k_h / 2 + 1),
                0..(k_w / 2 + 1),
                1..(max_stride_x + 1),
                1..(max_stride_y + 1),
                Just(w_in),
                Just(h_in),
                Just(c_in),
            )
        });
        Strategy::prop_map(
            strat,
            move |(
                kernel,
                data,
                bias,
                pad_y_0,
                pad_y_1,
                pad_x_0,
                pad_x_1,
                stride_x,
                stride_y,
                w_in,
                h_in,
                c_in,
            )| {
                let input_shape = TensorShape::new(vec![None, Some(h_in), Some(w_in), Some(c_in)]);
                let padding = ((pad_y_0, pad_y_1), (pad_x_0, pad_x_1));
                let strides = (stride_y, stride_x, 1);
                let conv_layer = Conv::new(kernel, bias, input_shape, strides, padding);
                (conv_layer, data)
            },
        )
    }

    proptest! {
        #[test]
        fn test_conv_equality((conv_layer, data) in conv_test_inputs(7, 7, 28, 28, 10, 10, 3, 3)) {
            let h_in = conv_layer.input_shape()[1].unwrap();
            let w_in = conv_layer.input_shape()[2].unwrap();
            let c_in = conv_layer.input_shape()[3].unwrap();

            let input_dims = h_in * w_in * c_in;
            let flat_data = data.view().into_shape(input_dims).unwrap();
            let output_shape = conv_layer.output_shape();
            let output_dims =
                output_shape[1].unwrap() * output_shape[2].unwrap() * output_shape[3].unwrap();

            let convolve_result = conv_layer.convolve(data.view());
            let affine = conv_layer.get_affine();
            let affine_result = affine.apply(&flat_data);

            let flat_convolve_result = convolve_result.into_shape(output_dims).unwrap();
            prop_assert!(
                flat_convolve_result.abs_diff_eq(&affine_result, 1e-10),
                "Unequal results. Convolve: {:?} Affine: {:?}, Affine mtx: {:?}, Flat data: {:?}, Output Shape: {:?}",
                flat_convolve_result,
                affine_result,
                affine,
                flat_data,
                output_shape,
            );
        }
    }
}

#![cfg(test)]
use crate::affine::Affine2;
use crate::Bounds;
use crate::Bounds1;
use crate::Layer;
use crate::DNN;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Zip;
use proptest::arbitrary::functor::ArbitraryF1;
use proptest::prelude::any;
use proptest::prelude::*;
use proptest::sample::SizeRange;
use std::mem;

prop_compose! {
    pub fn array1(len: usize)(v in Vec::lift1_with(-10. .. 10., SizeRange::new(len..=len))) -> Array1<f64> {
        Array1::from_vec(v)
    }
}

prop_compose! {
    pub fn array2(rows: usize, cols: usize)(v in Vec::lift1_with(array1(cols), SizeRange::new(rows..=rows))) -> Array2<f64> {
        assert!(rows > 0);
        ndarray::stack(Axis(0), &v.iter().map(|x| x.view()).collect::<Vec<ArrayView1<f64>>>()).unwrap()
    }
}

prop_compose! {
    pub fn affine2(in_dim: usize, out_dim: usize)(basis in array2(out_dim, in_dim), shift in array1(out_dim)) -> Affine2<f64> {
        Affine2::new(basis, shift)
    }
}

prop_compose! {
    pub fn bounds1(len: usize)(mut lower in array1(len), mut upper in array1(len)) -> Bounds1<f64> {
        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| if *l > *u {mem::swap(l, u)});
        assert!(Zip::from(&lower).and(&upper).all(|l, u| l <= u));
        Bounds::new(lower, upper)
    }
}

prop_compose! {
    pub fn bounds1_sample(bounds: Bounds1<f64>)(seed in any::<u64>()) -> Array1<f64> {
        bounds.sample_uniform(seed)
    }
}

prop_compose! {
    pub fn fc_dnn(input_size: usize, output_size: usize, nlayers: usize, max_layer_width: usize)(repr_sizes in Vec::lift1_with(1..max_layer_width, SizeRange::new(nlayers..=nlayers)).prop_map(move |mut x| {x.insert(0, input_size); x.push(output_size); x}))(affines in {let pairs = repr_sizes.iter().zip(repr_sizes.iter().skip(1)); pairs.map(|(&x, &y)| affine2(x,y)).collect::<Vec<_>>()}) -> DNN<f64> {
        let mut dnn = DNN::default();
        affines.into_iter().for_each(|aff| {
            let output_dim = aff.output_dim();
            dnn.add_layer(Layer::new_dense(aff));
            dnn.add_layer(Layer::new_relu(output_dim))
        });
        dnn
    }
}

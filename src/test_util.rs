#![cfg(test)]
use crate::affine::Affine2;
use crate::Affine;
use crate::Bounds;
use crate::Bounds1;
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
    pub fn array1(len: usize)(v in Vec::lift1_with(any::<f32>(), SizeRange::new(len..=len))) -> Array1<f32> {
        Array1::from_vec(v)
    }
}

prop_compose! {
    pub fn array2(rows: usize, cols: usize)(v in Vec::lift1_with(array1(cols), SizeRange::new(rows..=rows))) -> Array2<f32> {
        ndarray::stack(Axis(0), &v.iter().map(|x| x.view()).collect::<Vec<ArrayView1<f32>>>()).unwrap()
    }
}

prop_compose! {
    pub fn affine2(in_dim: usize, out_dim: usize)(basis in array2(out_dim, in_dim), shift in array1(out_dim)) -> Affine2<f32> {
        Affine2::new(basis, shift)
    }
}

prop_compose! {
    pub fn bounds1()(len in usize::MIN..100)(lower in array1(len), upper in array1(len)) -> Bounds1<f32> {
        Zip::from(&lower).and(&upper).for_each(|mut l, mut u| if l > u {mem::swap(&mut l, &mut u)});
        Bounds::new(lower, upper)
    }
}

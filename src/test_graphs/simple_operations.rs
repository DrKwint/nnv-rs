#![cfg(test)]

use crate::bounds::Bounds1;
use crate::graph::Operation;
use crate::NNVFloat;
use crate::{affine::Affine2, star::Star2};
use ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::{any::Any, fmt};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct SimpleAdd {}

#[typetag::serde]
impl Operation for SimpleAdd {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        vec![input
            .into_iter()
            .fold(Array::zeros(input[0].raw_dim()), |acc, &x| acc + x)]
    }

    fn forward2(&self, _input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
        todo!()
    }

    fn apply_bounds(
        &self,
        _bounds: &[Bounds1],
        _lower_aff: &[Affine2],
        _upper_aff: &[Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        todo!()
    }

    fn forward_star(
        &self,
        _stars: Vec<&Star2>,
        _activation_idx: Option<usize>,
        _parent_axis_aligned_input_bounds: Vec<&Bounds1>,
    ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
        todo!()
    }
}

impl fmt::Display for SimpleAdd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimpleAdd")
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct SimpleMultiply {}

#[typetag::serde]
impl Operation for SimpleMultiply {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        vec![input
            .into_iter()
            .fold(Array::ones(input[0].raw_dim()), |acc, &x| acc * x)]
    }

    fn forward2(&self, _input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
        todo!()
    }

    fn apply_bounds(
        &self,
        _bounds: &[Bounds1],
        _lower_aff: &[Affine2],
        _upper_aff: &[Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        todo!()
    }

    fn forward_star(
        &self,
        _stars: Vec<&Star2>,
        _activation_idx: Option<usize>,
        _parent_bounds: Vec<&Bounds1>,
    ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
        todo!()
    }
}

impl fmt::Display for SimpleMultiply {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimpleMultiply")
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct SimpleSquare {}

#[typetag::serde]
impl Operation for SimpleSquare {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        input.into_iter().map(|&x| x * x).collect()
    }

    fn forward2(&self, _input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
        todo!()
    }

    fn apply_bounds(
        &self,
        _bounds: &[Bounds1],
        _lower_aff: &[Affine2],
        _upper_aff: &[Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        todo!()
    }

    fn forward_star(
        &self,
        _stars: Vec<&Star2>,
        _activation_idx: Option<usize>,
        _parent_bounds: Vec<&Bounds1>,
    ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
        todo!()
    }
}

impl fmt::Display for SimpleSquare {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimpleSquare")
    }
}

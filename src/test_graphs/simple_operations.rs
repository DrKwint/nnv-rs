#![cfg(test)]

use crate::bounds::Bounds1;
use crate::graph::Operation;
use crate::NNVFloat;
use crate::{affine::Affine2, star::Star2};
use ndarray::{Array, Array1, Array2};
use std::ops::Deref;
use std::{any::Any, fmt};

#[derive(Default, Clone, Debug)]
pub struct SimpleAdd {}

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
        _bounds: &[&Bounds1],
        _lower_aff: &[&Affine2],
        _upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        todo!()
    }

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        _parent_stars: Vec<StarRef>,
        _step_id: Option<usize>,
        _input_bounds: &Bounds1,
        _parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>> {
        todo!()
    }
}

impl fmt::Display for SimpleAdd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimpleAdd")
    }
}

#[derive(Default, Clone, Debug)]
pub struct SimpleMultiply {}

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
        _bounds: &[&Bounds1],
        _lower_aff: &[&Affine2],
        _upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        todo!()
    }

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        _parent_stars: Vec<StarRef>,
        _step_id: Option<usize>,
        _input_bounds: &Bounds1,
        _parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>> {
        todo!()
    }
}

impl fmt::Display for SimpleMultiply {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimpleMultiply")
    }
}

#[derive(Default, Clone, Debug)]
pub struct SimpleSquare {}

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
        _bounds: &[&Bounds1],
        _lower_aff: &[&Affine2],
        _upper_aff: &[&Affine2],
    ) -> Vec<(Bounds1, Affine2, Affine2)> {
        todo!()
    }

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        _parent_stars: Vec<StarRef>,
        _step_id: Option<usize>,
        _input_bounds: &Bounds1,
        _parent_local_output_bounds_opt: Option<Vec<Bounds1Ref>>,
    ) -> Vec<Vec<(Star2, Option<Bounds1>)>> {
        todo!()
    }
}

impl fmt::Display for SimpleSquare {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimpleSquare")
    }
}

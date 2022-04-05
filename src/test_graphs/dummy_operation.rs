#![cfg(test)]

use crate::graph::Operation;
use crate::star::Star2;
use crate::{affine::Affine2, bounds::Bounds1, graph::OperationId, NNVFloat};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use std::{any::Any, fmt};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct DummyOperation {
    op_id: OperationId,
}

impl DummyOperation {
    pub fn new(op_id: OperationId) -> Self {
        Self { op_id }
    }

    pub fn get_op_id(&self) -> OperationId {
        self.op_id
    }
}

impl Operation for DummyOperation {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward1(&self, _input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
        todo!()
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

    fn forward_star<StarRef: Deref<Target = Star2>, Bounds1Ref: Deref<Target = Bounds1>>(
        &self,
        _parent_stars: Vec<StarRef>,
        _step_id: Option<usize>,
        _parent_axis_aligned_input_bounds: Vec<Bounds1Ref>,
    ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
        todo!()
    }
}

impl fmt::Display for DummyOperation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DummyOp")
    }
}

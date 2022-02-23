use crate::NNVFloat;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Dropout {
    prob: NNVFloat,
}

impl Dropout {
    pub fn new(prob: NNVFloat) -> Self {
        prob 
    }
}
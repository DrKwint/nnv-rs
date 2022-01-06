use std::time::Duration;

use rand::Rng;

use crate::NNVFloat;
use ndarray::Array1;

trait AdversarialStarSet {
    fn minimal_norm_adv_delta(&self, datum: &Array1<NNVFloat>) -> Array1<NNVFloat>;
    fn minimal_norm_adv_delta_target_class(&self, datum: &Array1<NNVFloat>) -> Array1<NNVFloat>;
}

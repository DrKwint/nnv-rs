use crate::{starsets::StarSet2, NNVFloat};
use ndarray::Array1;

pub trait AdversarialStarSet2: StarSet2 {
    // fn minimal_norm_attack_delta(&self, datum: &Array1<NNVFloat>) -> Array1<NNVFloat>;
    fn minimal_norm_targeted_attack_delta(
        &mut self,
        datum: &Array1<NNVFloat>,
        target_class_idx: usize,
    ) -> Array1<NNVFloat> {
        let input_leaf_node_id = self.run_datum_to_leaf(datum);
        let output_shape = self.get_dnn().output_shape();
        let nclasses = output_shape[-1].unwrap();
        let reachable_classes: Vec<usize> = (0..nclasses)
            .filter(|class| self.can_node_maximize_output_idx(input_leaf_node_id, *class))
            .collect();
        Array1::zeros(10)
    }
}

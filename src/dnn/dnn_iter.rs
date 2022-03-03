use super::dnn::DNN;
use crate::dnn::layer::Layer;
use log::trace;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter, Result};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
/// Indicated which operations have already been run
///
/// State  1     2    3
/// layer  None  Some Some
/// remain None  None Some
///
/// 1: Initial state
/// {1,2,3} -> 2: reach non-activation layer
/// {1,2} -> 3: reach activation layer
/// 3 -> 3: continue activation layer
pub struct DNNIndex {
    pub layer: Option<usize>,
    pub remaining_steps: Option<usize>,
}

impl DNNIndex {
    pub fn get_layer_idx(&self) -> usize {
        self.layer.map_or(0, |x| x + 1)
    }

    pub fn get_remaining_steps(&self) -> Option<usize> {
        self.remaining_steps
    }

    fn increment(&mut self, dnn: &DNN) {
        // Decrement active relu
        let mut advance_layer_flag = false;
        if let Some(ref mut step) = self.remaining_steps {
            match step {
                0 => advance_layer_flag = true,
                x => *x -= 1,
            }
        } else {
            advance_layer_flag = true;
        }

        // advance layer at the end of running a full layer (e.g., all step relus)
        if advance_layer_flag {
            if let Some(ref mut layer) = self.layer {
                *layer += 1;
            } else {
                self.layer = Some(0);
            }

            self.remaining_steps = if let Some(layer) = dnn.get_layer(self.layer.unwrap()) {
                if layer.is_activation() {
                    Some(layer.output_dims() - 1)
                } else {
                    None
                }
            } else {
                None
            };
        }
    }
}

#[derive(Debug, Clone)]
pub struct DNNIterator<'a> {
    dnn: &'a DNN,
    idx: DNNIndex,
    finished: bool,
}

impl<'a> Display for DNNIterator<'a> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "DNNIterator(finished: {}, idx_layer: {:?}, idx_remaining_steps {:?})",
            self.finished, self.idx.layer, self.idx.remaining_steps
        )
    }
}

impl<'a> DNNIterator<'a> {
    pub const fn new(dnn: &'a DNN, idx: DNNIndex) -> Self {
        Self {
            dnn,
            idx,
            finished: false,
        }
    }

    pub const fn get_idx(&self) -> DNNIndex {
        self.idx
    }

    pub fn get_layer(&self, idx: &DNNIndex) -> Option<&Box<dyn Layer>> {
        if let Some(layer_idx) = idx.layer {
            self.dnn.get_layer(layer_idx)
        } else {
            None
        }
    }
}

impl<'a> Iterator for DNNIterator<'a> {
    type Item = DNNIndex;

    fn next(&mut self) -> Option<Self::Item> {
        trace!("dnn iterator idx {:?}", self.get_idx());
        // Iterator base case
        if self.finished {
            return None;
        }

        // Increment index
        self.idx.increment(self.dnn); // Advance idx SM
        let layer_idx = self.idx.layer.unwrap(); // safe unwrap because idx SM is not in state 1
        if self.dnn.get_layer(layer_idx).is_some() {
            Some(self.idx)
        } else {
            self.finished = true;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::relu::ReLU;
    use super::*;
    use crate::starsets::StarSet;
    use crate::test_util::{asterism, fc_dnn};
    use proptest::prelude::*;

    #[test]
    fn test_dnn_index_increment() {
        let mut dnn = DNN::default();
        dnn.add_layer(Box::new(ReLU::new(4)));

        let mut idx = DNNIndex {
            layer: None,
            remaining_steps: None,
        };

        for i in (0..4).rev() {
            idx.increment(&dnn);
            assert_eq!(idx.layer.unwrap(), 0);
            assert_eq!(idx.remaining_steps.unwrap(), i);
        }

        idx.increment(&dnn);
        debug_assert!(dnn.get_layer(idx.layer.unwrap()).is_none());
    }

    proptest! {
        #[test]
        fn test_dnn_iterator_is_finite(dnn in fc_dnn(2, 2, 1, 2)) {
            let expected_steps: usize = dnn.get_layers().iter().enumerate().map(|(_, layer)| {
                if layer.is_activation() {
                    layer.output_dims()
                } else {
                    1
                }
            }).sum();

            let iter = DNNIterator::new(&dnn, DNNIndex{layer: None, remaining_steps: None});
            assert_eq!(iter.count(), expected_steps);
        }

        #[test]
        fn test_iter_all_layers(asterism in asterism(2, 2, 4, 2)) {
            let root_dnn_index = asterism.get_node_dnn_index(asterism.get_root_id());
            println!("root_dnn_index {:?}", root_dnn_index);
            let mut dnn_iter = DNNIterator::new(asterism.get_dnn(), root_dnn_index);
            println!("Start test");
            for layer in asterism.get_dnn().get_layers() {
                let iter_idx = dnn_iter.next();
                println!("iter_idx {:?}", iter_idx);
                let iter_layer = asterism.get_dnn().get_layer(iter_idx.unwrap().layer.unwrap()).unwrap();
                prop_assert_eq!(format!("{:?}", layer), format!("{:?}", iter_layer), "layer: {:?} vs iter_layer: {:?} at idx {:?}", layer, iter_layer, iter_idx);
            }
        }
    }
}

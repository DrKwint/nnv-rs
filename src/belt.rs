use crate::constellation::Constellation;
use crate::num::One;
use crate::NNVFloat;
use ndarray::{Array1, Dimension, Ix2};
use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};
use std::collections::BinaryHeap;

/// Frontier contains pairs of weight and index into the constellation arena
#[derive(Debug)]
struct Belt<'a, D: Dimension> {
    constellation: &'a mut Constellation<D>,
    frontier: BinaryHeap<(OrderedFloat<NNVFloat>, usize)>, // Tuples are ordered lexicographically
    leaves: Vec<(OrderedFloat<NNVFloat>, usize)>,
}

impl<'a, D: Dimension> Belt<'a, D> {
    pub fn new(constellation: &'a mut Constellation<D>) -> Self {
        let root_id = constellation.get_root_id();
        Self {
            constellation,
            frontier: BinaryHeap::from(vec![(OrderedFloat::one(), root_id)]),
            leaves: Vec::new(),
        }
    }
}

impl<'a> Belt<'a, Ix2> {
    /// Expansion criteria: in importance sampling, it is optimal to choose $q$
    /// that maximizes $p|f|$. So in this first iteration, our strategy will be
    /// to choose a weighting that overapproximates by using the cdf for $p$ as
    /// the proportional part of the cdf we know, and using the upper bounds of
    /// the absolute values of $f$ to approximate for the product's second term
    /// We'll always choose the node with greatest weight as the next to expand
    ///
    /// I'm so meta, even this acronym -XKCD
    pub fn expand<R: Rng>(&mut self, rng: &mut R, stability_eps: NNVFloat) -> bool {
        if let Some((_weight, next)) = self.frontier.pop() {
            let children = self.constellation.get_node_child_ids(next, stability_eps);
            for idx in children {
                let cdf = self
                    .constellation
                    .get_node_cdf(idx, 10, 10, rng, stability_eps);
                let bounds = self.constellation.get_node_output_bounds(idx);
                let f = bounds.0.abs().max(bounds.1.abs()); // Estimate to be refined
                let entry = (std::convert::From::from(f * cdf), idx);
                if self.constellation.is_node_leaf(idx, stability_eps) {
                    self.leaves.push(entry);
                } else {
                    self.frontier.push(entry);
                }
            }
            true
        } else {
            false
        }
    }

    /// Args:
    ///
    ///
    /// Returns:
    ///     `E_{N(mu, sigma)}[f]` where `f` is the underlying DNN, i.e. the expected value of the output
    pub fn importance_sample(&mut self, n_samples: NNVFloat, stability_eps: NNVFloat) -> NNVFloat {
        let mut rng = thread_rng();
        let total_weight: NNVFloat = self
            .frontier
            .iter()
            .map(|(weight, _idx)| weight)
            .sum::<OrderedFloat<NNVFloat>>()
            .0;
        self.frontier
            .clone()
            .iter()
            .chain(self.leaves.clone().iter())
            .map(|(weight, idx)| {
                let star_prob = weight.0 / total_weight;
                let n_local_samples: usize = (star_prob * n_samples) as usize;
                let local_samples: Vec<Array1<NNVFloat>> = self
                    .constellation
                    .sample_gaussian_node_output(*idx, &mut rng, n_local_samples, 10, stability_eps)
                    .into_iter()
                    .collect();
                let local_sum: NNVFloat = local_samples.iter().map(|x| x[[0]]).sum();
                let local_mean: NNVFloat = local_sum / (local_samples.len() as NNVFloat);
                local_mean * weight.0
            })
            .sum::<NNVFloat>()
            / total_weight
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::*;
    use proptest::*;

    proptest! {
        #[test]
        fn test_perform_importance_sample(mut constellation in constellation(2, 1, 2, 2)) {
            let mut belt = Belt::new(&mut constellation);
            belt.importance_sample(1.0, 1e-4);
        }

        #[test]
        fn test_multiple_beltloops(mut constellation in constellation(2, 2, 2, 2)) {
            // let mut rng = rand::thread_rng();
            // let mut belt = Belt::new(&mut constellation);
            // (0..10).for_each(|_| {
            //     belt.expand(&mut rng, 1e-4);
            //     belt.importance_sample(100., 1e-4);
            // });
        }
    }
}

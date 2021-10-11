use crate::constellation::Constellation;
use crate::NNVFloat;
use ndarray::Dimension;
use ndarray::Ix2;
use ordered_float::OrderedFloat;
use rand::thread_rng;
use std::collections::BinaryHeap;
use std::convert::TryInto;

/// Frontier contains pairs of weight and index into the constellation arena
struct Belt<'a, T: NNVFloat, D: Dimension> {
    constellation: &'a mut Constellation<T, D>,
    frontier: BinaryHeap<(OrderedFloat<T>, usize)>, // Tuples are ordered lexicographically
    leaves: Vec<(OrderedFloat<T>, usize)>,
}

impl<'a, T: NNVFloat, D: Dimension> Belt<'a, T, D> {
    pub fn new(constellation: &'a mut Constellation<T, D>) -> Self {
        let root_id = constellation.get_root_id();
        Self {
            constellation,
            frontier: BinaryHeap::from(vec![(OrderedFloat::one(), root_id)]),
            leaves: Vec::new(),
        }
    }
}

impl<'a, T: crate::NNVFloat> Belt<'a, T, Ix2> {
    /// Expansion criteria: in importance sampling, it is optimal to choose $q$
    /// that maximizes $p|f|$. So in this first iteration, our strategy will be
    /// to choose a weighting that overapproximates by using the cdf for $p$ as
    /// the proportional part of the cdf we know, and using the upper bounds of
    /// the absolute values of $f$ to approximate for the product's second term
    /// We'll always choose the node with greatest weight as the next to expand
    ///
    /// I'm so meta, even this acronym -XKCD
    pub fn expand(&mut self) {
        let (weight, next) = self.frontier.pop().unwrap();
        let children = self.constellation.get_node_child_ids(next);
        children.into_iter().for_each(|idx| {
            let cdf = self.constellation.get_node_cdf(idx, 10, 10);
            let bounds = self.constellation.get_node_output_bounds(idx);
            let f = bounds.0.abs().max(bounds.1.abs()); // Estimate to be refined
            let entry = (std::convert::From::from(f * cdf), idx);
            if self.constellation.is_node_leaf(idx) {
                self.leaves.push(entry);
            } else {
                self.frontier.push(entry);
            }
        });
    }

    /// Args:
    ///
    ///
    /// Returns:
    ///     E_{N(mu, sigma)}[f] where f is the underlying DNN, i.e. the expected value of the output
    pub fn importance_sample(&self, n_samples: T) {
        //-> T {
        let mut rng = thread_rng();
        let total_weight: T = self
            .frontier
            .iter()
            .map(|(weight, _idx)| weight)
            .sum::<OrderedFloat<T>>()
            .0;
        let samples = self
            .frontier
            .iter()
            .chain(self.leaves.iter())
            .map(|(weight, idx)| {
                let star_prob = weight.0 / total_weight;
                let n_local_samples = ((star_prob * n_samples).into() as u64).try_into().unwrap();
                self.constellation
                    .sample_gaussian_node(*idx, &mut rng, n_local_samples, 10)
            });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::*;
    use proptest::*;

    proptest! {
        #[test]
        fn test_perform_importance_sample(mut constellation in generic_constellation(2, 2, 2, 2)) {
            let mut belt = Belt::new(&mut constellation);
            belt.expand();
            belt.importance_sample(1.0);
        }
    }
}

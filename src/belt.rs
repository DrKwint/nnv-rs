use crate::constellation::Constellation;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Dimension;
use ndarray::Ix2;
use ordered_float::OrderedFloat;
use rand::thread_rng;
use rand::Rng;
use std::collections::BinaryHeap;
use std::convert::TryInto;

/// Frontier contains pairs of weight and index into the constellation arena
#[derive(Debug)]
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
    pub fn expand<R: Rng>(&mut self, rng: &mut R) -> bool {
        if let Some((_weight, next)) = self.frontier.pop() {
            let children = self.constellation.get_node_child_ids(next);
            children.into_iter().for_each(|idx| {
                let cdf = self.constellation.get_node_cdf(idx, 10, 10, rng);
                let bounds = self.constellation.get_node_output_bounds(idx);
                let f = bounds.0.abs().max(bounds.1.abs()); // Estimate to be refined
                let entry = (std::convert::From::from(f * cdf), idx);
                if self.constellation.is_node_leaf(idx) {
                    self.leaves.push(entry);
                } else {
                    self.frontier.push(entry);
                }
            });
            true
        } else {
            false
        }
    }

    /// Args:
    ///
    ///
    /// Returns:
    ///     E_{N(mu, sigma)}[f] where f is the underlying DNN, i.e. the expected value of the output
    pub fn importance_sample(&mut self, n_samples: T) -> T {
        let mut rng = thread_rng();
        let total_weight: T = self
            .frontier
            .iter()
            .map(|(weight, _idx)| weight)
            .sum::<OrderedFloat<T>>()
            .0;
        self.frontier
            .clone()
            .iter()
            .chain(self.leaves.clone().iter())
            .map(|(weight, idx)| {
                let star_prob = weight.0 / total_weight;
                let n_local_samples = ((star_prob * n_samples).into() as u64).try_into().unwrap();
                let local_samples: Vec<Array1<T>> = self
                    .constellation
                    .sample_gaussian_node_output(*idx, &mut rng, n_local_samples, 10)
                    .into_iter()
                    .collect();
                let local_sum: T = local_samples.iter().map(|x| x[[0]]).sum();
                let local_mean: T = local_sum / (local_samples.len() as f64).into();
                local_mean * weight.0
            })
            .sum::<T>()
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
            belt.importance_sample(1.0);
        }

        #[test]
        fn test_multiple_beltloops(mut constellation in constellation(2, 2, 2, 2)) {
            let mut rng = rand::thread_rng();
            let mut belt = Belt::new(&mut constellation);
            (0..10).for_each(|_| {
                belt.expand(&mut rng);
                println!("Belt: {:?}", belt);
                belt.importance_sample(100.);
            });
        }
    }
}

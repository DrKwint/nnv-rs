use crate::bounds::Bounds1;
use crate::constellation::Constellation;
use crate::probstarset::ProbStarSet2;
use crate::star_node::StarNodeType;
use crate::starset::StarSet;
use crate::starset::StarSet2;
use crate::util::FstOrdTuple;
use crate::NNVFloat;
use log::{debug, info};
use ndarray::{concatenate, Array1, Axis, Dimension, Ix2};
use num::Float;
use num::{One, Zero};
use ordered_float::OrderedFloat;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::iter;
use std::ops::Neg;
use std::time::{Duration, Instant};

pub struct Asterism<'a, D: Dimension> {
    constellation: &'a mut Constellation<D>,
    safe_value: NNVFloat,
    is_feasible: Vec<bool>,
}

#[derive(Serialize, Deserialize)]
struct SerializableAsterism<D: Dimension> {
    constellation: Constellation<D>,
    safe_value: NNVFloat,
    is_feasible: Vec<bool>,
}

impl<'a, D: Dimension> SerializableAsterism<D> {
    fn get_asterism(&'a mut self) -> Asterism<'a, D> {
        Asterism {
            constellation: &mut self.constellation,
            safe_value: self.safe_value,
            is_feasible: self.is_feasible.clone(),
        }
    }
}

impl<'a, D: Dimension> From<Asterism<'a, D>> for SerializableAsterism<D> {
    fn from(item: Asterism<'a, D>) -> Self {
        Self {
            constellation: item.constellation.clone(),
            safe_value: item.safe_value,
            is_feasible: item.is_feasible,
        }
    }
}

impl<'a, D: Dimension> Asterism<'a, D> {
    pub fn new(constellation: &'a mut Constellation<D>, safe_value: NNVFloat) -> Self {
        Self {
            constellation,
            safe_value,
            is_feasible: vec![true],
        }
    }

    fn get_feasible(&self, id: usize) -> bool {
        *self.is_feasible.get(id).or(Some(&true)).unwrap()
    }

    fn set_feasible(&mut self, id: usize, val: bool) {
        if id >= self.is_feasible.len() {
            self.is_feasible.resize(id + 1, true);
        }
        self.is_feasible[id] = val;
    }
}

impl<'a> Asterism<'a, Ix2> {
    pub fn get_node_output_bounds(&mut self, node_id: usize) -> (NNVFloat, NNVFloat) {
        let (node_mut, loc, scale, dnn) = self.constellation.get_node_mut_with_borrows(node_id);
        node_mut.get_output_bounds(dnn, &|x| (x.lower()[[0]], x.upper()[[0]]))
    }

    /// # Panics
    pub fn dfs_samples<R: Rng>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
        time_limit_opt: Option<Duration>,
    ) {
        let samples = self.constellation.sample_root_node(num_samples, rng);
        let activation_patterns = self
            .constellation
            .get_dnn()
            .calculate_activation_pattern(samples);
        // currently, tilting is propagated, so we need to initialize it for the root node
        self.constellation
            .get_node_gaussian_distribution(self.constellation.get_root_id());
        // For each datum
        for i in 0..num_samples {
            // Start at the root
            let mut current_node_id = self.constellation.get_root_id();
            let mut current_node_type = self.constellation.get_node_type(current_node_id).clone();
            // For each ReLU layer activation pattern
            for layer_activations in &activation_patterns {
                // Go through the Affine
                if let StarNodeType::Affine { child_idx } = current_node_type {
                    current_node_id = child_idx;
                    current_node_type = self.constellation.get_node_type(current_node_id).clone();
                }
                // For each activation
                for activation in layer_activations.row(i) {
                    // Estimate output bounds and potentially stop
                    let current_output_bounds = self.get_node_output_bounds(current_node_id);
                    println!(
                        "Node {} output bounds: {:?}",
                        current_node_id, current_output_bounds
                    );
                    // Select a child node based on the activation
                    if let StarNodeType::StepRelu {
                        dim,
                        fst_child_idx,
                        snd_child_idx,
                    } = current_node_type
                    {
                        if *activation {
                            current_node_id = fst_child_idx;
                        } else {
                            current_node_id =
                                snd_child_idx.expect("Error selecting a second child!");
                        }
                    } else {
                        panic!("Expected a ReLU layer!");
                    }
                }
            }
        }
    }

    pub fn get_overapproximated_infeasible_input_regions(&self) -> Vec<Bounds1> {
        let unsafe_nodes: HashSet<usize> = self
            .is_feasible
            .iter()
            .enumerate()
            .filter(|(_pos, &x)| !x)
            .map(|(pos, _x)| pos)
            .collect();
        let deduped_unsafe_nodes = unsafe_nodes.iter().filter(|idx| {
            self.constellation
                .get_node_ancestors(**idx)
                .iter()
                .all(|x| !unsafe_nodes.contains(x))
        });
        deduped_unsafe_nodes
            .filter_map(|idx| self.constellation.get_node_input_bounds(*idx))
            .cloned()
            .collect()
    }

    /// Sample from a Gaussian distribution with an upper bound on the output value
    ///
    /// This function uses a two step sampling process. It first walks the Constellation's binary
    /// tree, which is defined by the parameters of the underlying neural network. This walk
    /// continues until a safe star has been reached, and then it's sampled in the second step to
    /// produce the final set of samples.
    ///
    /// # Arguments
    ///
    /// * `safe_value` - Maximum value an output can have to be considered safe
    /// * `rng` - A random number generator
    /// * `cdf_samples` - Number of samples to use in estimating a polytope's CDF under the Gaussian
    /// * `num_samples` - Number of samples to return from the final star that's sampled
    ///
    /// # Panics
    ///
    /// # Returns
    ///
    /// (Vec of samples, log path probability, total invalid cdf proportion)
    ///
    /// Sample probability can be calculated (in a network without dropouts) by calculating
    /// gaussian probability and dividing by (1 - invalid cdf proportion)
    #[allow(clippy::too_many_lines)]
    pub fn sample_safe_star<R: Rng>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
        time_limit_opt: Option<Duration>,
    ) -> Option<(Vec<Array1<NNVFloat>>, NNVFloat, NNVFloat)> {
        let start_time = Instant::now();
        let cdf_samples = self.constellation.get_cdf_samples();
        let max_iters = self.constellation.get_max_accept_reject_iters();
        let stability_eps = self.constellation.get_stability_eps();
        let mut sample_heap: BinaryHeap<FstOrdTuple<Reverse<OrderedFloat<f64>>, _>> =
            BinaryHeap::new();
        let mut current_node = 0;
        let mut path = vec![];
        let mut path_logp = 0.;
        let mut total_infeasible_cdf = 0.;
        let infeasible_reset = |me: &mut Self,
                                x: usize,
                                path: &mut Vec<usize>,
                                path_logp: &mut NNVFloat,
                                total_infeasible_cdf: &mut NNVFloat,
                                rng: &mut R|
         -> usize {
            info!("Infeasible reset at depth {}!", path.len());
            me.set_feasible(x, false);
            let mut infeas_cdf = me.constellation.get_node_cdf(x, rng);
            path.drain(..).rev().for_each(|x| {
                // check if all chilren are infeasible
                if me
                    .constellation
                    .get_node_child_ids(x)
                    .iter()
                    .any(|&x| me.get_feasible(x))
                {
                    // if not infeasible, update CDF
                    me.constellation
                        .add_node_cdf(x, NNVFloat::neg(NNVFloat::one()) * infeas_cdf);
                } else {
                    me.set_feasible(x, false);
                    infeas_cdf = me.constellation.get_node_cdf(x, rng);
                }
            });
            *total_infeasible_cdf += infeas_cdf;
            *path_logp = NNVFloat::zero();
            0
        };
        loop {
            debug!("Current node: {:?}", current_node);
            // base case for feasability
            if current_node == 0 && !self.get_feasible(current_node) {
                info!("No feasible value exists!");
                return None;
            }
            // Try to sample and return if a valid sample is found
            let safe_sample_opt = self.try_safe_sample(current_node, rng, num_samples);
            debug_assert!(self
                .constellation
                .try_get_node_gaussian_distribution(current_node)
                .is_some());
            match safe_sample_opt {
                Ok(safe_samples) => {
                    info!(
                        "Safe samples found less than {} at depth {}",
                        self.safe_value,
                        path.len()
                    );
                    return Some((safe_samples, path_logp, total_infeasible_cdf));
                }
                Err(sample_val_pairs) => sample_val_pairs
                    .into_iter()
                    .map(|(sample, val)| (Reverse(OrderedFloat(val)), sample.mapv(OrderedFloat)))
                    .for_each(|pair| sample_heap.push(FstOrdTuple(pair))),
            };
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds = if current_node == 0 {
                    (NNVFloat::neg_infinity(), NNVFloat::infinity())
                } else {
                    self.get_node_output_bounds(current_node)
                };
                debug!("Output bounds: {:?}", output_bounds);
                debug_assert!(self
                    .constellation
                    .try_get_node_gaussian_distribution(current_node)
                    .is_some());
                if output_bounds.1 <= self.safe_value {
                    // handle case where star is safe
                    info!(
                        "Safe sample with value at most {} at depth {} (bounds {:?})",
                        self.safe_value,
                        path.len(),
                        output_bounds
                    );
                    let safe_sample =
                        self.constellation
                            .sample_gaussian_node(current_node, rng, num_samples);
                    return Some((safe_sample, path_logp, total_infeasible_cdf));
                } else if output_bounds.0 > self.safe_value {
                    // handle case where star is infeasible
                    self.set_feasible(current_node, false);
                    current_node = infeasible_reset(
                        self,
                        current_node,
                        &mut path,
                        &mut path_logp,
                        &mut total_infeasible_cdf,
                        rng,
                    );
                    continue;
                } else if let StarNodeType::Leaf = self.constellation.get_node_type(current_node) {
                    // do procedure to select safe part
                    info!(
                        "Safe sample with value at most {} at leaf (bounds {:?})",
                        self.safe_value, output_bounds
                    );
                    let safe_sample = self.constellation.sample_gaussian_node_safe(
                        current_node,
                        rng,
                        num_samples,
                        max_iters,
                        self.safe_value,
                        stability_eps,
                    );
                    return Some((safe_sample, path_logp, total_infeasible_cdf));
                }
                // otherwise, push to path and continue expanding
                path.push(current_node);
            }
            // check timeout
            if let Some(time_limit) = time_limit_opt {
                if start_time.elapsed() >= time_limit {
                    info!("Unsafe sample after timeout at depth {}", path.len());
                    let samples = sample_heap
                        .into_iter_sorted()
                        .take(num_samples)
                        .map(|pair| pair.0 .1.mapv(|x| x.0))
                        .collect();
                    return Some((samples, path_logp, total_infeasible_cdf));
                }
            }
            debug_assert!(self
                .constellation
                .try_get_node_gaussian_distribution(current_node)
                .is_some());
            // expand node
            {
                let result = self.select_safe_child(current_node, path_logp, rng);
                match result {
                    Some((node, logp)) => {
                        debug_assert!(node != current_node);
                        current_node = node;
                        path_logp += logp;
                    }
                    None => {
                        current_node = infeasible_reset(
                            self,
                            current_node,
                            &mut path,
                            &mut path_logp,
                            &mut total_infeasible_cdf,
                            rng,
                        );
                    }
                }
            }
        }
    }

    fn try_safe_sample<R: Rng>(
        &mut self,
        current_node: usize,
        rng: &mut R,
        num_samples: usize,
    ) -> Result<Vec<Array1<NNVFloat>>, Vec<(Array1<NNVFloat>, NNVFloat)>> {
        let unsafe_sample = self
            .constellation
            .sample_gaussian_node(current_node, rng, num_samples);
        let vals: Vec<_> = {
            let sample_iter = unsafe_sample.iter();
            let fixed_input_part: Option<Array1<NNVFloat>> = {
                let unsafe_len = unsafe_sample[0].len();
                self.constellation
                    .get_node_input_bounds(current_node)
                    .map(|bounds| {
                        let fixed_bounds: Bounds1 = bounds.split_at(bounds.ndim() - unsafe_len).0;
                        let fixed_array: Array1<NNVFloat> = fixed_bounds.lower().to_owned();
                        fixed_array
                    })
            };
            if let Some(fix_part) = fixed_input_part {
                sample_iter
                    .zip(iter::repeat(fix_part))
                    .map(|(unfix, fix)| concatenate(Axis(0), &[fix.view(), unfix.view()]).unwrap())
                    .map(|x| self.constellation.get_dnn().forward1(x))
                    .collect()
            } else {
                sample_iter
                    .map(|x| self.constellation.get_dnn().forward1(x.clone()))
                    .collect()
            }
        };
        let safe_subset: Vec<_> = unsafe_sample
            .iter()
            .zip(vals.iter())
            .filter(|(_sample, out)| out[[0]] < self.safe_value)
            .map(|(sample, _val)| sample.to_owned())
            .collect();
        if safe_subset.is_empty() {
            Err(unsafe_sample
                .into_iter()
                .zip(vals.iter())
                .map(|(x, out)| (x.to_owned(), out[[0]]))
                .collect())
        } else {
            Ok(safe_subset)
        }
    }

    /// Given a node, samples one of its children and returns the log probability
    ///
    /// Returns:
    ///     `child_idx` (usize): The child that was reached in the sampling procedure
    ///     `path_logp` (`NNVFloat`): The log probability of reaching the child from the root node
    ///
    /// # Panics
    fn select_safe_child<R: Rng>(
        &mut self,
        current_node: usize,
        mut path_logp: NNVFloat,
        rng: &mut R,
    ) -> Option<(usize, NNVFloat)> {
        debug_assert!(self
            .constellation
            .try_get_node_gaussian_distribution(current_node)
            .is_some());
        let current_node_type = self.constellation.get_node_type(current_node);
        match *current_node_type {
            // leaf node, which must be partially safe and partially unsafe
            StarNodeType::Leaf => {
                panic!();
            }
            StarNodeType::Affine { child_idx } => Some((child_idx, path_logp)),
            StarNodeType::StepRelu {
                fst_child_idx,
                snd_child_idx,
                ..
            } => {
                let mut children = vec![fst_child_idx];
                if let Some(snd_child) = snd_child_idx {
                    children.push(snd_child);
                }

                children = children
                    .into_iter()
                    .filter(|&child| self.get_feasible(child))
                    .collect();
                match children.len() {
                    0 => None,
                    1 => Some((children[0], path_logp)),
                    2 => {
                        let fst_cdf = self.constellation.get_node_cdf(children[0], rng);
                        let snd_cdf = self.constellation.get_node_cdf(children[1], rng);
                        debug!(
                            "Selecting between 2 children with CDFs: {} and {}",
                            fst_cdf, snd_cdf
                        );
                        debug!(
                            "Selecting between 2 children with CDFs: {} and {}",
                            fst_cdf, snd_cdf
                        );

                        // Handle the case where a CDF gives a non-normal value
                        debug_assert!(fst_cdf.is_sign_positive());
                        debug_assert!(snd_cdf.is_sign_positive());
                        let fst_prob = match (fst_cdf.is_normal(), snd_cdf.is_normal()) {
                            (true, true) => fst_cdf / (fst_cdf + snd_cdf),
                            (false, true) => {
                                let parent_cdf = self.constellation.get_node_cdf(current_node, rng);
                                let mut derived_fst_cdf = parent_cdf - snd_cdf;

                                // CDFs are estimates and it is
                                // possible for the parent to have
                                // smaller CDF than the child.
                                if derived_fst_cdf.is_sign_negative() {
                                    derived_fst_cdf = NNVFloat::epsilon();
                                }
                                self.constellation
                                    .set_node_cdf(children[0], derived_fst_cdf);
                                derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                            }
                            (true, false) => {
                                let parent_cdf = self.constellation.get_node_cdf(current_node, rng);
                                let mut derived_snd_cdf = parent_cdf - fst_cdf;
                                if derived_snd_cdf.is_sign_negative() {
                                    derived_snd_cdf = NNVFloat::epsilon();
                                }
                                self.constellation
                                    .set_node_cdf(children[1], derived_snd_cdf);
                                fst_cdf / (fst_cdf + derived_snd_cdf)
                            }
                            // If both CDFs are non-normal, we'll mark the current node as infeasible
                            (false, false) => {
                                return None;
                            }
                        };
                        // Safe unwrap due to above handling
                        let dist = Bernoulli::new(fst_prob).unwrap_or_else(|_| panic!("Bad fst_prob: {:?} fst_cdf: {:?} snd_cdf: {:?}, fst_n {:?}, snd_n {:?}", fst_prob, fst_cdf, snd_cdf, fst_cdf.is_normal(), snd_cdf.is_normal()));
                        if dist.sample(rng) {
                            path_logp += fst_prob.ln();
                            Some((children[0], path_logp))
                        } else {
                            path_logp += (NNVFloat::one() - fst_prob).ln();
                            Some((children[1], path_logp))
                        }
                    }
                    _ => {
                        panic!();
                    }
                }
            }
            StarNodeType::StepReluDropOut {
                dropout_prob,
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
                ..
            } => {
                let dropout_dist = Bernoulli::new(dropout_prob).unwrap();
                if dropout_dist.sample(rng) {
                    path_logp += dropout_prob.ln();
                    Some((fst_child_idx, path_logp))
                } else {
                    path_logp += (NNVFloat::one() - dropout_prob).ln();

                    let mut children = vec![];

                    if let Some(snd_child) = snd_child_idx {
                        children.push(snd_child);
                    }
                    if let Some(trd_child) = trd_child_idx {
                        children.push(trd_child);
                    }
                    children = children
                        .into_iter()
                        .filter(|&child| self.get_feasible(child))
                        .collect();
                    match children.len() {
                        0 => None,
                        1 => Some((children[0], path_logp)),
                        _ => {
                            let fst_cdf = self.constellation.get_node_cdf(children[0], rng);
                            let snd_cdf = self.constellation.get_node_cdf(children[1], rng);

                            // Handle the case where a CDF gives a non-normal value
                            let fst_prob = match (fst_cdf.is_normal(), snd_cdf.is_normal()) {
                                (true, true) => fst_cdf / (fst_cdf + snd_cdf),
                                (false, true) => {
                                    let parent_cdf =
                                        self.constellation.get_node_cdf(current_node, rng);
                                    let derived_fst_cdf = parent_cdf - snd_cdf;
                                    self.constellation
                                        .set_node_cdf(children[0], derived_fst_cdf);
                                    derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                                }
                                (true, false) => {
                                    let parent_cdf =
                                        self.constellation.get_node_cdf(current_node, rng);
                                    let derived_snd_cdf = parent_cdf - fst_cdf;
                                    self.constellation
                                        .set_node_cdf(children[1], derived_snd_cdf);
                                    fst_cdf / (fst_cdf + derived_snd_cdf)
                                }
                                // If both CDFs are non-normal, we'll mark the current node as infeasible
                                (false, false) => {
                                    return None;
                                }
                            };
                            // Safe unwrap due to above handling
                            let dist = Bernoulli::new(fst_prob).unwrap();
                            if dist.sample(rng) {
                                path_logp += fst_prob.ln();
                                Some((children[0], path_logp))
                            } else {
                                path_logp += (NNVFloat::one() - fst_prob).ln();
                                Some((children[1], path_logp))
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::*;
    use proptest::*;

    proptest! {
        #[test]
        fn test_sample_safe_star(mut constellation in generic_constellation(2, 2, 2, 2)) {
            let mut rng = rand::thread_rng();
            let mut asterism = Asterism::new(&mut constellation, 1.);
            let default: Array1<f64> = Array1::zeros(asterism.constellation.get_dnn().input_shape()[0].unwrap());
            let sample = asterism.sample_safe_star(1, &mut rng, 1, 1, None, 1e-4);
        }

        #[test]
        fn test_dfs_samples(mut constellation in generic_constellation(2, 2, 2, 2)) {
                let num_samples = 4;
                let cdf_samples = 100;
                let max_iters = 10;
                let time_limit_opt = None;
                let stability_eps = 1e-10;

                let mut rng = rand::thread_rng();
                let mut asterism = Asterism::new(&mut constellation, 1.);
                asterism.dfs_samples(num_samples, &mut rng, cdf_samples, max_iters, time_limit_opt, stability_eps);
            }
    }
}

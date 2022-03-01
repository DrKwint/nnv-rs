use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Neg;
use std::time::Duration;
use std::time::Instant;

use crate::bounds::Bounds1;
use crate::num::Float;
use crate::num::One;
use crate::num::Zero;
use crate::rand::distributions::Distribution;
use crate::star_node::StarNodeType;
use crate::starsets::ProbStarSet;
use crate::starsets::ProbStarSet2;
use crate::util::FstOrdTuple;
use crate::NNVFloat;
use log::debug;
use log::info;
use ndarray::concatenate;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::Slice;
use ordered_float::OrderedFloat;
use rand::distributions::Bernoulli;
use rand::Rng;
use std::iter;

pub trait CensoredProbStarSet<D: 'static + Dimension>: ProbStarSet<D> {
    type I<'a>: Iterator<Item = &'a Option<bool>>
    where
        Self: 'a,
        D: 'a;

    fn get_safe_value(&self) -> NNVFloat;
    fn set_safe_value(&mut self, val: NNVFloat);
    fn get_feasible_iter(&self) -> Self::I<'_>;
    fn is_node_infeasible(&self, id: usize) -> bool;
    fn get_node_feasibility(&self, id: usize) -> Option<bool>;
    fn set_node_feasibility(&mut self, id: usize, val: bool);
}

pub trait CensoredProbStarSet2: CensoredProbStarSet<Ix2> + ProbStarSet2 {
    fn get_node_output_bounds(&mut self, node_id: usize) -> (NNVFloat, NNVFloat) {
        let outer_bounds: Bounds1 = self.get_input_bounds().as_ref().cloned().unwrap();
        let (node_mut, loc, scale, dnn) = self.get_node_mut_with_borrows(node_id);
        node_mut.get_output_bounds(dnn, &|x| (x.lower()[[0]], x.upper()[[0]]), &outer_bounds)
    }

    /// # Panics
    fn dfs_samples<R: Rng>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
        time_limit_opt: Option<Duration>,
    ) {
        let samples = {
            let without_fixed = self.sample_root_node(num_samples, rng);
            let almost_samples = if let Some(bounds) = self.get_input_bounds() {
                let unfixed_dims = without_fixed.shape()[0]; // Assuming rank 2 here
                let mut fixed = bounds
                    .fixed_vals_or_zeros()
                    .slice_axis(Axis(0), Slice::from(..-(unfixed_dims as isize)))
                    .insert_axis(Axis(1))
                    .to_owned();
                if fixed.iter().any(|x| *x != 0.) {
                    let zeros: Array2<f64> = Array2::zeros((1, 1000));
                    fixed = fixed + zeros;
                    fixed
                        .append(Axis(0), without_fixed.view())
                        .expect("Could not append!");
                    fixed
                } else {
                    without_fixed
                }
            } else {
                without_fixed
            };
            almost_samples
        };
        let activation_patterns = self.get_dnn().calculate_activation_pattern2(samples);
        // currently, tilting is propagated, so we need to initialize it for the root node
        self.get_node_gaussian_distribution(self.get_root_id());

        // For each datum
        for i in 0..num_samples {
            // Start at the root
            let mut current_node_id = self.get_root_id();
            let mut current_node_type = self.get_node_type(current_node_id).clone();

            // For each ReLU layer activation pattern
            for layer_activations in &activation_patterns {
                // Go through the Affine
                if let StarNodeType::Affine { child_idx } = current_node_type {
                    current_node_id = child_idx;
                    current_node_type = self.get_node_type(current_node_id).clone();
                }
                // For each activation
                for activation in layer_activations.column(i) {
                    // Estimate output bounds and potentially stop
                    // TODO
                    let current_output_bounds = self.get_node_output_bounds(current_node_id);
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
                        current_node_type = self.get_node_type(current_node_id).clone();
                    } else {
                        panic!("Expected a ReLU layer! Found {:?}", current_node_type);
                    }
                }
            }
        }
    }

    fn get_overapproximated_infeasible_input_regions(&self) -> Vec<Bounds1> {
        let unsafe_nodes: Vec<usize> = self
            .get_feasible_iter()
            .enumerate()
            .filter(|(_pos, x)| !matches!(x, Some(true)))
            .map(|(pos, _x)| pos)
            .collect();
        let deduped_unsafe_nodes = unsafe_nodes.iter().filter(|idx| {
            self.get_node_ancestors(**idx)
                .iter()
                .all(|x| *x == **idx || !unsafe_nodes.contains(x))
        });
        deduped_unsafe_nodes
            .filter_map(|idx| {
                self.get_node(*idx)
                    .try_get_axis_aligned_input_bounds()
                    .as_ref()
            })
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
    fn sample_safe_star<R: Rng>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
        time_limit_opt: Option<Duration>,
    ) -> Option<(Vec<Array1<NNVFloat>>, NNVFloat, NNVFloat)> {
        let start_time = Instant::now();
        let cdf_samples = self.get_cdf_samples();
        let max_iters = self.get_max_accept_reject_iters();
        let stability_eps = self.get_stability_eps();
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
            me.set_node_feasibility(x, false);
            let mut infeas_cdf = me.get_node_cdf(x, rng);
            path.drain(..).rev().for_each(|x| {
                // check if all chilren are infeasible
                if me
                    .get_node_child_ids(x)
                    .iter()
                    .any(|&x| !me.is_node_infeasible(x))
                {
                    // if not infeasible, update CDF
                    me.add_node_cdf(x, NNVFloat::neg(NNVFloat::one()) * infeas_cdf);
                } else {
                    me.set_node_feasibility(x, false);
                    infeas_cdf = me.get_node_cdf(x, rng);
                }
            });
            *total_infeasible_cdf += infeas_cdf;
            *path_logp = NNVFloat::zero();
            0
        };
        loop {
            info!("Current node: {:?}", current_node);
            info!(
                "Current idx: {:?}",
                self.get_node(current_node).get_dnn_index()
            );
            // base case for feasability
            if current_node == 0 && self.is_node_infeasible(current_node) {
                info!("No feasible value exists!");
                return None;
            }
            // Try to sample and return if a valid sample is found
            let safe_sample_opt = self.try_safe_sample(current_node, rng, num_samples);
            debug_assert!(self
                .try_get_node_gaussian_distribution(current_node)
                .is_some());
            match safe_sample_opt {
                Ok(safe_samples) => {
                    info!(
                        "Safe samples found less than {} at depth {}",
                        self.get_safe_value(),
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
                    .try_get_node_gaussian_distribution(current_node)
                    .is_some());
                if output_bounds.1 <= self.get_safe_value() {
                    // handle case where star is safe
                    info!(
                        "Safe sample with value at most {} at depth {} (bounds {:?})",
                        self.get_safe_value(),
                        path.len(),
                        output_bounds
                    );
                    let safe_sample = self.sample_gaussian_node(current_node, rng, num_samples);
                    return Some((safe_sample, path_logp, total_infeasible_cdf));
                } else if output_bounds.0 > self.get_safe_value() {
                    // handle case where star is infeasible
                    self.set_node_feasibility(current_node, false);
                    current_node = infeasible_reset(
                        self,
                        current_node,
                        &mut path,
                        &mut path_logp,
                        &mut total_infeasible_cdf,
                        rng,
                    );
                    continue;
                } else if let StarNodeType::Leaf = self.get_node_type(current_node) {
                    // do procedure to select safe part
                    info!(
                        "Safe sample with value at most {} at leaf (bounds {:?})",
                        self.get_safe_value(),
                        output_bounds
                    );
                    let safe_sample = self.sample_gaussian_node_safe(
                        current_node,
                        rng,
                        num_samples,
                        max_iters,
                        self.get_safe_value(),
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

    /// # Errors
    fn try_safe_sample<R: Rng>(
        &mut self,
        current_node: usize,
        rng: &mut R,
        num_samples: usize,
    ) -> Result<Vec<Array1<NNVFloat>>, Vec<(Array1<NNVFloat>, NNVFloat)>> {
        let unsafe_sample = self.sample_gaussian_node(current_node, rng, num_samples);
        let vals: Vec<_> = {
            let sample_iter = unsafe_sample.iter();
            let fixed_input_part: Option<Array1<NNVFloat>> = {
                let unsafe_len = unsafe_sample[0].len();
                self.get_input_bounds().as_ref().map(|bounds| {
                    let fixed_bounds: Bounds1 = bounds.split_at(bounds.ndim() - unsafe_len).0;
                    let fixed_array: Array1<NNVFloat> = fixed_bounds.lower().to_owned();
                    fixed_array
                })
            };
            if let Some(fix_part) = fixed_input_part {
                sample_iter
                    .zip(iter::repeat(fix_part))
                    .map(|(unfix, fix)| concatenate(Axis(0), &[fix.view(), unfix.view()]).unwrap())
                    .map(|x| self.get_dnn().forward1(x))
                    .collect()
            } else {
                sample_iter
                    .map(|x| self.get_dnn().forward1(x.clone()))
                    .collect()
            }
        };
        let safe_subset: Vec<_> = unsafe_sample
            .iter()
            .zip(vals.iter())
            .filter(|(_sample, out)| out[[0]] < self.get_safe_value())
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
            .try_get_node_gaussian_distribution(current_node)
            .is_some());
        let current_node_type = self.get_node_type(current_node);
        match *current_node_type {
            // leaf node, which must be partially safe and partially unsafe
            StarNodeType::Leaf => {
                panic!();
            }
            StarNodeType::Affine { child_idx } => Some((child_idx, path_logp)),
            StarNodeType::Conv { child_idx } => Some((child_idx, path_logp)),
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
                    .filter(|&child| !self.is_node_infeasible(child))
                    .collect();
                match children.len() {
                    0 => None,
                    1 => Some((children[0], path_logp)),
                    2 => {
                        let fst_cdf = self.get_node_cdf(children[0], rng);
                        let snd_cdf = self.get_node_cdf(children[1], rng);
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
                                let parent_cdf = self.get_node_cdf(current_node, rng);
                                let mut derived_fst_cdf = parent_cdf - snd_cdf;

                                // CDFs are estimates and it is
                                // possible for the parent to have
                                // smaller CDF than the child.
                                if derived_fst_cdf.is_sign_negative() {
                                    derived_fst_cdf = NNVFloat::epsilon();
                                }
                                self.set_node_cdf(children[0], derived_fst_cdf);
                                derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                            }
                            (true, false) => {
                                let parent_cdf = self.get_node_cdf(current_node, rng);
                                let mut derived_snd_cdf = parent_cdf - fst_cdf;
                                if derived_snd_cdf.is_sign_negative() {
                                    derived_snd_cdf = NNVFloat::epsilon();
                                }
                                self.set_node_cdf(children[1], derived_snd_cdf);
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
                        .filter(|&child| !self.is_node_infeasible(child))
                        .collect();
                    match children.len() {
                        0 => None,
                        1 => Some((children[0], path_logp)),
                        _ => {
                            let fst_cdf = self.get_node_cdf(children[0], rng);
                            let snd_cdf = self.get_node_cdf(children[1], rng);

                            // Handle the case where a CDF gives a non-normal value
                            let fst_prob = match (fst_cdf.is_normal(), snd_cdf.is_normal()) {
                                (true, true) => fst_cdf / (fst_cdf + snd_cdf),
                                (false, true) => {
                                    let parent_cdf = self.get_node_cdf(current_node, rng);
                                    let derived_fst_cdf = parent_cdf - snd_cdf;
                                    self.set_node_cdf(children[0], derived_fst_cdf);
                                    derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                                }
                                (true, false) => {
                                    let parent_cdf = self.get_node_cdf(current_node, rng);
                                    let derived_snd_cdf = parent_cdf - fst_cdf;
                                    self.set_node_cdf(children[1], derived_snd_cdf);
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

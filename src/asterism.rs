use crate::bounds::Bounds1;
use crate::constellation::Constellation;
use crate::star_node::StarNodeType;
use crate::util::diag_gaussian_accept_reject;
use crate::util::gaussian_logp;
use crate::NNVFloat;
use log::{debug, info};
use ndarray::{concatenate, s, Array1, Axis, Dimension, Ix2};
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use std::iter;
use std::time::{Duration, Instant};

pub struct Asterism<'a, T: NNVFloat, D: Dimension> {
    constellation: &'a mut Constellation<T, D>,
    safe_value: T,
    is_feasible: Vec<bool>,
}

impl<'a, T: NNVFloat, D: Dimension> Asterism<'a, T, D> {
    pub fn new(constellation: &'a mut Constellation<T, D>, safe_value: T) -> Self {
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

impl<'a, T: NNVFloat> Asterism<'a, T, Ix2> {
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
        cdf_samples: usize,
        max_iters: usize,
        time_limit_opt: Option<Duration>,
        stability_eps: T,
    ) -> Option<(Vec<Array1<T>>, T, T)> {
        let start_time = Instant::now();
        let mut best_sample = None;
        let mut best_sample_val = T::infinity();
        let mut current_node = 0;
        let mut path = vec![];
        let mut path_logp = T::zero();
        let mut total_infeasible_cdf = T::zero();
        let infeasible_reset = |me: &mut Self,
                                x: usize,
                                path: &mut Vec<usize>,
                                path_logp: &mut T,
                                total_infeasible_cdf: &mut T,
                                rng: &mut R|
         -> usize {
            info!("Infeasible reset at depth {}!", path.len());
            me.set_feasible(x, false);
            let mut infeas_cdf =
                me.constellation
                    .get_node_cdf(x, cdf_samples, max_iters, rng, stability_eps);
            path.drain(..).rev().for_each(|x| {
                // check if all chilren are infeasible
                if me
                    .constellation
                    .get_node_child_ids(x, stability_eps)
                    .iter()
                    .any(|&x| me.get_feasible(x))
                {
                    // if not infeasible, update CDF
                    me.constellation
                        .add_node_cdf(x, T::neg(T::one()) * infeas_cdf);
                } else {
                    me.set_feasible(x, false);
                    infeas_cdf = me.constellation.get_node_cdf(
                        x,
                        cdf_samples,
                        max_iters,
                        rng,
                        stability_eps,
                    );
                }
            });
            *total_infeasible_cdf += infeas_cdf;
            *path_logp = T::zero();
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
            let safe_sample_opt =
                self.try_safe_sample(current_node, rng, num_samples, max_iters, stability_eps);
            debug_assert!(self
                .constellation
                .try_get_gaussian_distribution(current_node)
                .is_some());
            match safe_sample_opt {
                Ok((safe_sample, est_cost)) => {
                    info!(
                        "Safe sample with value {} less than {} at depth {}",
                        est_cost,
                        self.safe_value,
                        path.len()
                    );
                    return Some((vec![safe_sample], path_logp, total_infeasible_cdf));
                }
                Err(Some((unsafe_sample, val))) => {
                    if val < best_sample_val {
                        best_sample = Some(unsafe_sample);
                        best_sample_val = val;
                    }
                }
                Err(None) => {}
            };
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds = if current_node == 0 {
                    (T::neg_infinity(), T::infinity())
                } else {
                    self.constellation.get_node_output_bounds(current_node)
                };
                debug!("Output bounds: {:?}", output_bounds);
                debug_assert!(self
                    .constellation
                    .try_get_gaussian_distribution(current_node)
                    .is_some());
                if output_bounds.1 <= self.safe_value {
                    // handle case where star is safe
                    info!(
                        "Safe sample with value at most {} at depth {} (bounds {:?})",
                        self.safe_value,
                        path.len(),
                        output_bounds
                    );
                    let safe_sample = self.constellation.sample_gaussian_node(
                        current_node,
                        rng,
                        num_samples,
                        max_iters,
                        stability_eps,
                    );
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
                } else if let StarNodeType::Leaf = self
                    .constellation
                    .get_node_type(current_node, stability_eps)
                {
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
                    info!(
                        "Unsafe sample after timeout with value {:?} at depth {}",
                        best_sample_val,
                        path.len()
                    );
                    return best_sample.map(|x| (vec![x], path_logp, total_infeasible_cdf));
                }
            }
            debug_assert!(self
                .constellation
                .try_get_gaussian_distribution(current_node)
                .is_some());
            // expand node
            {
                let result = self.select_safe_child(
                    current_node,
                    path_logp,
                    cdf_samples,
                    max_iters,
                    rng,
                    stability_eps,
                );
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
        max_iters: usize,
        stability_eps: T,
    ) -> Result<(Array1<T>, T), Option<(Array1<T>, T)>> {
        let unsafe_sample = self.constellation.sample_gaussian_node(
            current_node,
            rng,
            num_samples,
            max_iters,
            stability_eps,
        );
        let input_set = self
            .constellation
            .get_node_reduced_input_polytope(current_node);
        let unsafe_len = unsafe_sample[0].len();
        let fixed_input_part: Option<Array1<T>> = self
            .constellation
            .get_node_input_bounds(current_node)
            .map(|bounds| {
                let fixed_bounds: Bounds1<T> = bounds.split_at(bounds.ndim() - unsafe_len).0;
                let fixed_array: Array1<T> = fixed_bounds.lower().to_owned();
                fixed_array
            });
        let mut rng = rand::thread_rng();
        let mut best_sample = None;
        let mut best_val = T::infinity();
        let sample_opt: Option<(Array1<T>, T)> = if let Some(fixed_input_part) = fixed_input_part {
            let mut input_iter = unsafe_sample
                .into_iter()
                .filter(|x| match &input_set {
                    Some(poly) => poly.is_member(&x.view()),
                    None => true,
                })
                .filter(|x| {
                    /*
                    diag_gaussian_accept_reject(
                        &x.mapv(|x| x.into()).view(),
                        &self.constellation.get_loc().mapv(|x| x.into()).view(),
                        &self
                            .constellation
                            .get_scale()
                            .diag()
                            .mapv(|x| x.into())
                            .view(),
                        &mut rng,
                    )
                    */
                    let logp = gaussian_logp(
                        &x.mapv(|x| x.into()).view(),
                        &self.constellation.get_loc().mapv(|x| x.into()).view(),
                        &self
                            .constellation
                            .get_scale()
                            .diag()
                            .mapv(|x| x.into())
                            .view(),
                    );
                    logp > -2.5
                })
                .zip(iter::repeat(fixed_input_part))
                .map(|(unfix, fix)| concatenate(Axis(0), &[fix.view(), unfix.view()]).unwrap());
            input_iter.find_map(|x| {
                let output = self.constellation.get_dnn().forward1(x.clone());
                debug_assert_eq!(output.len(), 1);
                if output[[0]] < self.safe_value {
                    Some((x, output[[0]]))
                } else {
                    if output[[0]] < best_val {
                        best_sample = Some(x);
                        best_val = output[[0]];
                    }
                    None
                }
            })
        } else {
            let mut input_iter = unsafe_sample.into_iter();
            input_iter.find_map(|x| {
                let output = self.constellation.get_dnn().forward1(x.clone());
                debug_assert_eq!(output.len(), 1);
                if output[[0]] < self.safe_value {
                    Some((x, output[[0]]))
                } else {
                    if output[[0]] < best_val {
                        best_sample = Some(x);
                        best_val = output[[0]];
                    }
                    None
                }
            })
        };
        sample_opt
            .map(|(sample, est_cost)| {
                let unfixed = sample.slice(s![-(unsafe_len as isize)..]).to_owned();
                (unfixed, est_cost)
            })
            .ok_or_else(|| {
                (best_sample.map(|x| (x.slice(s![-(unsafe_len as isize)..]).to_owned(), best_val)))
            })
    }

    /// Given a node, samples one of its children and returns the log probability
    ///
    /// Returns:
    ///     `child_idx` (usize): The child that was reached in the sampling procedure
    ///     `path_logp` (T): The log probability of reaching the child from the root node
    ///
    /// # Panics
    fn select_safe_child<R: Rng>(
        &mut self,
        current_node: usize,
        mut path_logp: T,
        cdf_samples: usize,
        max_iters: usize,
        rng: &mut R,
        stability_eps: T,
    ) -> Option<(usize, T)> {
        debug_assert!(self
            .constellation
            .try_get_gaussian_distribution(current_node)
            .is_some());
        let current_node_type = self
            .constellation
            .get_node_type(current_node, stability_eps);
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
                        let fst_cdf = self.constellation.get_node_cdf(
                            children[0],
                            cdf_samples * 4,
                            max_iters,
                            rng,
                            stability_eps,
                        );
                        let snd_cdf = self.constellation.get_node_cdf(
                            children[1],
                            cdf_samples,
                            max_iters,
                            rng,
                            stability_eps,
                        );
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
                                let parent_cdf = self.constellation.get_node_cdf(
                                    current_node,
                                    cdf_samples,
                                    max_iters,
                                    rng,
                                    stability_eps,
                                );
                                let mut derived_fst_cdf = parent_cdf - snd_cdf;

                                // CDFs are estimates and it is
                                // possible for the parent to have
                                // smaller CDF than the child.
                                if derived_fst_cdf.is_sign_negative() {
                                    derived_fst_cdf = T::epsilon();
                                }
                                self.constellation
                                    .set_node_cdf(children[0], derived_fst_cdf);
                                derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                            }
                            (true, false) => {
                                let parent_cdf = self.constellation.get_node_cdf(
                                    current_node,
                                    cdf_samples,
                                    max_iters,
                                    rng,
                                    stability_eps,
                                );
                                let mut derived_snd_cdf = parent_cdf - fst_cdf;
                                if derived_snd_cdf.is_sign_negative() {
                                    derived_snd_cdf = T::epsilon();
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
                        let dist = Bernoulli::new(fst_prob.into()).unwrap_or_else(|_| panic!("Bad fst_prob: {:?} fst_cdf: {:?} snd_cdf: {:?}, fst_n {:?}, snd_n {:?}", fst_prob, fst_cdf, snd_cdf, fst_cdf.is_normal(), snd_cdf.is_normal()));
                        if dist.sample(rng) {
                            path_logp += fst_prob.ln();
                            Some((children[0], path_logp))
                        } else {
                            path_logp += (T::one() - fst_prob).ln();
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
                let dropout_dist = Bernoulli::new(dropout_prob.into()).unwrap();
                if dropout_dist.sample(rng) {
                    path_logp += dropout_prob.ln();
                    Some((fst_child_idx, path_logp))
                } else {
                    path_logp += (T::one() - dropout_prob).ln();

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
                            let fst_cdf = self.constellation.get_node_cdf(
                                children[0],
                                cdf_samples,
                                max_iters,
                                rng,
                                stability_eps,
                            );
                            let snd_cdf = self.constellation.get_node_cdf(
                                children[1],
                                cdf_samples,
                                max_iters,
                                rng,
                                stability_eps,
                            );

                            // Handle the case where a CDF gives a non-normal value
                            let fst_prob = match (fst_cdf.is_normal(), snd_cdf.is_normal()) {
                                (true, true) => fst_cdf / (fst_cdf + snd_cdf),
                                (false, true) => {
                                    let parent_cdf = self.constellation.get_node_cdf(
                                        current_node,
                                        cdf_samples,
                                        max_iters,
                                        rng,
                                        stability_eps,
                                    );
                                    let derived_fst_cdf = parent_cdf - snd_cdf;
                                    self.constellation
                                        .set_node_cdf(children[0], derived_fst_cdf);
                                    derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                                }
                                (true, false) => {
                                    let parent_cdf = self.constellation.get_node_cdf(
                                        current_node,
                                        cdf_samples,
                                        max_iters,
                                        rng,
                                        stability_eps,
                                    );
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
                            let dist = Bernoulli::new(fst_prob.into()).unwrap();
                            if dist.sample(rng) {
                                path_logp += fst_prob.ln();
                                Some((children[0], path_logp))
                            } else {
                                path_logp += (T::one() - fst_prob).ln();
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
    }
}

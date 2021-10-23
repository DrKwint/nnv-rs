use crate::constellation::Constellation;
use crate::star_node::StarNodeType;
use crate::Bounds1;
use crate::NNVFloat;
use log::{debug, info, trace};
use ndarray::concatenate;
use ndarray::s;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::Ix2;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use std::iter;

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
    /// (Vec of (samples, sample probabilities), branch probability)
    #[allow(clippy::too_many_lines)]
    pub fn sample_safe_star<R: Rng>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
        cdf_samples: usize,
        max_iters: usize,
    ) -> Option<(Vec<(Array1<T>, T)>, T)> {
        let mut current_node = 0;
        let mut path = vec![];
        let mut path_logp = T::zero();
        let infeasible_reset = |me: &mut Self,
                                x: usize,
                                path: &mut Vec<usize>,
                                path_logp: &mut T,
                                rng: &mut R|
         -> usize {
            debug!("Infeasible reset!");
            me.set_feasible(x, false);
            let mut infeas_cdf = me
                .constellation
                .get_node_cdf(x, cdf_samples, max_iters, rng);
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
                        .add_node_cdf(x, T::neg(T::one()) * infeas_cdf);
                } else {
                    me.set_feasible(x, false);
                    infeas_cdf = me
                        .constellation
                        .get_node_cdf(x, cdf_samples, max_iters, rng)
                }
            });
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
            let safe_sample_opt = self.try_safe_sample(current_node, rng, num_samples, max_iters);
            if let Some(safe_sample) = safe_sample_opt {
                info!(
                    "Random sample with value at most {} at depth {:?}",
                    self.safe_value,
                    path.len()
                );
                return Some((vec![safe_sample], path_logp));
            }
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds = self.constellation.get_node_output_bounds(current_node);
                debug!("Output bounds: {:?}", output_bounds);
                if output_bounds.1 <= self.safe_value {
                    // handle case where star is safe
                    info!(
                        "Safe sample with value at most {} at depth {} with bounds {:?}",
                        self.safe_value,
                        path.len(),
                        output_bounds
                    );
                    let safe_sample = self.constellation.sample_gaussian_node(
                        current_node,
                        rng,
                        num_samples,
                        max_iters,
                    );
                    return Some((safe_sample, path_logp));
                } else if output_bounds.0 > self.safe_value {
                    // handle case where star is infeasible
                    self.set_feasible(current_node, false);
                    current_node =
                        infeasible_reset(self, current_node, &mut path, &mut path_logp, rng);
                    continue;
                } else if let StarNodeType::Leaf = self.constellation.get_node_type(current_node) {
                    // do procedure to select safe part
                    info!("Safe sample at leaf with bounds {:?}", output_bounds);
                    let safe_sample = self.constellation.sample_gaussian_node_safe(
                        current_node,
                        rng,
                        num_samples,
                        max_iters,
                        self.safe_value,
                    );
                    return Some((safe_sample, path_logp));
                }
                // otherwise, push to path and continue expanding
                path.push(current_node);
            }
            // expand node
            {
                let result =
                    self.select_safe_child(current_node, path_logp, cdf_samples, max_iters, rng);
                match result {
                    Some((node, logp)) => {
                        current_node = node;
                        path_logp += logp;
                    }
                    None => {
                        current_node =
                            infeasible_reset(self, current_node, &mut path, &mut path_logp, rng);
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
    ) -> Option<(Array1<T>, T)> {
        let unsafe_sample =
            self.constellation
                .sample_gaussian_node(current_node, rng, num_samples, max_iters);
        let unsafe_len = unsafe_sample[0].0.len();
        let fixed_input_part: Option<Array1<T>> = {
            self.constellation
                .get_input_bounds()
                .as_ref()
                .map(|bounds| {
                    let fixed_bounds: Bounds1<T> = bounds.split_at(unsafe_len).1;
                    let fixed_array: Array1<T> = fixed_bounds.lower().to_owned();
                    fixed_array
                })
        };
        let dnn_idx = self.constellation.get_node_dnn_index(current_node);
        let sample_opt = if let Some(fixed_input_part) = fixed_input_part {
            let mut input_iter = unsafe_sample
                .into_iter()
                .zip(iter::repeat(fixed_input_part))
                .map(|((unfix, logp), fix)| {
                    (
                        concatenate(Axis(0), &[unfix.view(), fix.view()]).unwrap(),
                        logp,
                    )
                });
            input_iter.find_map(|(x, logp)| {
                let output = self.constellation.get_dnn().forward1(x.clone());
                debug_assert_eq!(output.len(), 1);
                if output[[0]] < self.safe_value {
                    Some((x, logp))
                } else {
                    None
                }
            })
        } else {
            let mut input_iter = unsafe_sample.into_iter();
            input_iter.find_map(|(x, logp)| {
                let output = self.constellation.get_dnn().forward1(x.clone());
                debug_assert_eq!(output.len(), 1);
                if output[[0]] < self.safe_value {
                    Some((x, logp))
                } else {
                    None
                }
            })
        };
        sample_opt.map(|(sample, logp)| {
            let unfixed = sample.slice(s![..unsafe_len]).to_owned();
            (unfixed, logp)
        })
    }

    /// Given a node, samples one of its children and returns the log probability
    ///
    /// Returns:
    ///     child_idx (usize): The child that was reached in the sampling procedure
    ///     path_logp (T): The log probability of reaching the child from the root node
    ///
    /// # Panics
    fn select_safe_child<R: Rng>(
        &mut self,
        current_node: usize,
        mut path_logp: T,
        cdf_samples: usize,
        max_iters: usize,
        rng: &mut R,
    ) -> Option<(usize, T)> {
        let input_bounds = self.constellation.get_input_bounds();
        let current_node_type = self.constellation.get_node_type(current_node);
        match *current_node_type {
            // leaf node, which must be partially safe and partially unsafe
            StarNodeType::Leaf => {
                panic!();
            }
            StarNodeType::Affine { child_idx } => {
                self.constellation.initialize_node_tilting_from_parent(
                    current_node,
                    child_idx,
                    max_iters,
                );
                Some((child_idx, path_logp))
            }
            StarNodeType::StepRelu {
                dim,
                fst_child_idx,
                snd_child_idx,
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
                    1 => {
                        self.constellation.initialize_node_tilting_from_parent(
                            current_node,
                            children[0],
                            max_iters,
                        );
                        Some((children[0], path_logp))
                    }
                    2 => {
                        self.constellation.initialize_node_tilting_from_parent(
                            current_node,
                            children[0],
                            max_iters,
                        );
                        self.constellation.initialize_node_tilting_from_parent(
                            current_node,
                            children[1],
                            max_iters,
                        );
                        let fst_cdf = self.constellation.get_node_cdf(
                            children[0],
                            cdf_samples,
                            max_iters,
                            rng,
                        );
                        let snd_cdf = self.constellation.get_node_cdf(
                            children[1],
                            cdf_samples,
                            max_iters,
                            rng,
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
                dim,
                dropout_prob,
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
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
                            );
                            let snd_cdf = self.constellation.get_node_cdf(
                                children[1],
                                cdf_samples,
                                max_iters,
                                rng,
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
                let default = Array1::zeros(asterism.constellation.get_dnn().input_shape()[0].unwrap());
                let sample = asterism.sample_safe_star(1, &mut rng, 1, 1).unwrap_or((vec![(default, 0.69)], 0.69));
    <<<<<<< HEAD
    =======
                println!("sample {:?}", sample);
    >>>>>>> main
                assert_eq!(sample.0[0].0.len(), asterism.constellation.get_dnn().input_shape()[0].unwrap(), "expected sample shape: {:?}", asterism.constellation.get_dnn().input_shape()[0].unwrap())
            }
        }
}

use crate::constellation::Constellation;
use crate::star_node::StarNodeType;
use crate::NNVFloat;
use log::debug;
use ndarray::Array1;
use ndarray::Dimension;
use ndarray::Ix2;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;

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
        debug!(
            "Running sample_safe_star with: safe_value {}",
            self.safe_value
        );
        let mut current_node = 0;
        let mut path = vec![];
        let mut path_logp = T::zero();
        let input_bounds = self.constellation.get_input_bounds().clone();
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
                return None;
            }
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds = self.constellation.get_node_output_bounds(current_node);
                debug!("Output bounds: {:?}", output_bounds);
                if output_bounds.1 <= self.safe_value {
                    // handle case where star is safe
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
        match *self.constellation.get_node_type(current_node) {
            // leaf node, which must be partially safe and partially unsafe
            StarNodeType::Leaf => {
                panic!();
            }
            StarNodeType::Affine { child_idx } => Some((child_idx, path_logp)),
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
                        /*
                        debug!(
                            "One child with cdf: {}",
                            self.arena[children[0]].gaussian_cdf(
                                &self.loc,
                                &self.scale,
                                cdf_samples,
                                max_iters,
                                &input_bounds
                            )
                        );
                        */
                        Some((children[0], path_logp))
                    }
                    2 => {
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
            println!("sample {:?}", sample);
            assert_eq!(sample.0[0].0.len(), asterism.constellation.get_dnn().input_shape()[0].unwrap(), "expected sample shape: {:?}", asterism.constellation.get_dnn().input_shape()[0].unwrap())
        }
    }
}

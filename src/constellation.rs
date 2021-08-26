//! Data structure representing the paths through a DNN with sets as input/output
use crate::dnn::DNNIndex;
use crate::dnn::DNNIterator;
use crate::star::Star;
use crate::util::*;
use crate::Bounds;
use crate::StarNode;
use crate::DNN;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use num::Float;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use std::iter::Sum;

/// Data structure representing the paths through a deep neural network (DNN)
pub struct Constellation<T: Float, D: Dimension> {
    arena: Vec<StarNode<T, D>>,
    root_id: usize,
    dnn: DNN<T>,
    input_bounds: Option<Bounds<T, D>>,
}

impl<T: Float, D: Dimension> Constellation<T, D>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::ops::MulAssign
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::AddAssign,
    f64: std::convert::From<T>,
{
    /// Instantiate a Constellation with given input set and network
    pub fn new(input_star: Star<T, D>, dnn: DNN<T>, input_bounds: Option<Bounds<T, D>>) -> Self {
        let star_node = StarNode {
            star: input_star,
            children: None,
            dnn_index: DNNIndex::default(),
            star_cdf: None,
            output_bounds: None,
            is_feasible: true,
        };
        // let mut arena = Arena::new();

        let mut arena = Vec::new();
        arena.push(star_node);
        let root_id = 0;
        Self {
            arena,
            root_id,
            dnn,
            input_bounds,
        }
    }
}

impl<T: Float> Constellation<T, Ix2>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::ops::MulAssign
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::AddAssign
        + Default
        + Sum,
    f64: std::convert::From<T>,
{
    fn expand_node(&mut self, id: usize) -> Vec<usize> {
        let node = &self.arena[id];
        let children = node.expand(&mut self.arena, &mut DNNIterator::from(self.dnn));
        children
    }

    /// Sample from a Gaussian distribution with an upper bound on the output value
    ///
    /// This function uses a two step sampling process. The first walks the Constellation's binary
    /// tree, which is defined by the parameters of the underlying neural network. This walk
    /// continues until a safe star has been reached, and then it's sampled in the second step to
    /// produce the final set of samples.
    ///
    /// # Arguments
    ///
    /// * `loc` - Gaussian location parameter
    /// * `scale` - Gaussian covariance matrix or, if diagonal, vector
    /// * `safe_value` - Maximum value an output can have to be considered safe
    /// * `cdf_samples` - Number of samples to use in estimating a polytope's CDF under the Gaussian
    /// * `num_samples` - Number of samples to return from the final star that's sampled
    ///
    /// # Panics
    ///
    /// # Returns
    ///
    /// (Vec of samples, Array of sample probabilities, branch probability)
    #[allow(clippy::too_many_arguments)]
    pub fn bounded_sample_multivariate_gaussian<R: Rng>(
        &mut self,
        rng: &mut R,
        loc: &Array1<T>,
        scale: &Array2<T>,
        safe_value: T,
        cdf_samples: usize,
        num_samples: usize,
        max_iters: usize,
    ) -> (Vec<(Array1<T>, T)>, T) {
        if let Some((safe_star, path_logp)) =
            self.sample_safe_star(loc, scale, safe_value, cdf_samples, max_iters)
        {
            (
                safe_star.gaussian_sample(
                    rng,
                    loc,
                    scale,
                    num_samples,
                    max_iters,
                    &self.input_bounds,
                ),
                path_logp,
            )
        } else {
            panic!()
        }
    }
}

impl<T: Float> Constellation<T, Ix2>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::ops::MulAssign
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::AddAssign
        + Default
        + Sum,
    f64: std::convert::From<T>,
{
    /// # Panics
    #[allow(clippy::too_many_lines)]
    pub fn sample_safe_star(
        &mut self,
        loc: &Array1<T>,
        scale: &Array2<T>,
        safe_value: T,
        cdf_samples: usize,
        max_iters: usize,
    ) -> Option<(StarNode<T, Ix2>, T)> {
        let mut rng = rand::thread_rng();
        let mut current_node = self.root_id;
        let mut path = vec![];
        let mut path_logp = T::zero();
        let root_id = self.root_id;
        let infeasible_reset = |arena: &mut Vec<StarNode<T, Ix2>>,
                                x: usize,
                                path: &mut Vec<usize>,
                                path_logp: &mut T|
         -> usize {
            arena[x].set_feasible(false);
            let mut infeas_cdf = arena[x].gaussian_cdf(loc, scale, cdf_samples, max_iters);
            path.drain(..).rev().for_each(|x| {
                // check if all chilren are infeasible
                if arena[x]
                    .get_child_ids()
                    .unwrap()
                    .iter()
                    .any(|x| arena[x].is_feasible)
                {
                    // if not infeasible, update CDF
                    arena[x].add_cdf(T::neg(T::one()) * infeas_cdf);
                } else {
                    arena[x].set_feasible(false);
                    infeas_cdf = arena[x].gaussian_cdf(loc, scale, cdf_samples, max_iters)
                }
            });
            *path_logp = T::zero();
            root_id
        };
        loop {
            // base case for feasability
            if current_node == self.root_id && !self.arena[current_node].get_feasible() {
                return None;
            }
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds =
                    self.arena[current_node]
                        .get_mut()
                        .get_output_bounds(&self.dnn, 0, &|x| (x.lower()[[0]], x.upper()[[0]]));
                if output_bounds.1 < safe_value {
                    // handle case where star is safe
                    let safe_star = self.arena[current_node].get().clone();
                    return Some((safe_star, path_logp));
                } else if output_bounds.0 > safe_value {
                    // handle case where star is infeasible
                    self.arena[current_node].get_mut().set_feasible(false);
                    infeasible_reset(&mut self.arena, current_node, &mut path, &mut path_logp);
                    continue;
                } else {
                    // otherwise, push to path and continue expanding
                    path.push(current_node);
                }
            }
            // expand node
            {
                let children: Vec<usize> = if self.arena[current_node].get().get_expanded() {
                    current_node.children(&self.arena).collect()
                } else {
                    self.arena[current_node].get_mut().set_expanded();
                    self.expand_node(current_node)
                };
                current_node = match children.len() {
                    // leaf node, which must be partially safe and partially unsafe
                    0 => {
                        let safe_star = self.arena[current_node].get().clone();
                        return Some((safe_star, path_logp));
                    }
                    1 => children[0],
                    2 => {
                        // check feasibility
                        match (
                            self.arena[children[0]].get().get_feasible(),
                            self.arena[children[1]].get().get_feasible(),
                        ) {
                            (false, false) => infeasible_reset(
                                &mut self.arena,
                                current_node,
                                &mut path,
                                &mut path_logp,
                            ),
                            (true, false) => children[0],
                            (false, true) => children[1],
                            (true, true) => {
                                let fst_cdf = self.arena[children[0]].get_mut().gaussian_cdf(
                                    loc,
                                    scale,
                                    cdf_samples,
                                    max_iters,
                                );
                                let snd_cdf = self.arena[children[1]].get_mut().gaussian_cdf(
                                    loc,
                                    scale,
                                    cdf_samples,
                                    max_iters,
                                );
                                // Handle the case where a CDF gives a non-normal value
                                let fst_prob = match (fst_cdf.is_normal(), snd_cdf.is_normal()) {
                                    (true, true) => fst_cdf / (fst_cdf + snd_cdf),
                                    (false, true) => {
                                        let parent_cdf = self.arena[current_node]
                                            .get_mut()
                                            .gaussian_cdf(loc, scale, cdf_samples, max_iters);
                                        let derived_fst_cdf = parent_cdf - snd_cdf;
                                        self.arena[children[0]].get_mut().set_cdf(derived_fst_cdf);
                                        derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                                    }
                                    (true, false) => {
                                        let parent_cdf = self.arena[current_node]
                                            .get_mut()
                                            .gaussian_cdf(loc, scale, cdf_samples, max_iters);
                                        let derived_snd_cdf = parent_cdf - fst_cdf;
                                        self.arena[children[1]].get_mut().set_cdf(derived_snd_cdf);
                                        fst_cdf / (fst_cdf + derived_snd_cdf)
                                    }
                                    // If both CDFs are non-normal, we'll mark the current node as infeasible
                                    (false, false) => {
                                        infeasible_reset(
                                            &mut self.arena,
                                            current_node,
                                            &mut path,
                                            &mut path_logp,
                                        );
                                        continue;
                                    }
                                };
                                // Safe unwrap due to above handling
                                let dist = Bernoulli::new(fst_prob.into()).unwrap();
                                if dist.sample(&mut rng) {
                                    path_logp += fst_prob.ln();
                                    children[0]
                                } else {
                                    path_logp += (T::one() - fst_prob).ln();
                                    children[1]
                                }
                            }
                        }
                    }
                    _ => panic!(),
                };
            }
        }
    }
}

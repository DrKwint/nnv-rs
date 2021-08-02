//! Data structure representing the paths through a DNN with sets as input/output
use crate::bounds::Bounds1;
use crate::star::Star;
use crate::Layer;
use crate::DNN;
use indextree::{Arena, NodeId};
use ndarray::IxDyn;
use ndarray::{Array1, Array2};
use num::Float;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;

/// Data structure representing the paths through a deep neural network (DNN)
pub struct Constellation<T: Float> {
    arena: Arena<StarNode<T>>,
    root: NodeId,
    dnn: DNN<T>,
    input_bounds: Option<Bounds1<T>>,
}

impl<T: Float> Constellation<T>
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
    pub fn new(input_star: Star<T, IxDyn>, dnn: DNN<T>, input_bounds: Option<Bounds1<T>>) -> Self {
        let star_node = StarNode {
            star: input_star,
            dnn_layer: 0,
            remaining_steps: None,
            cdf: None,
            output_bounds: None,
            is_expanded: false,
            is_feasible: true,
        };
        let mut arena = Arena::new();
        let root = arena.new_node(star_node);
        Self {
            arena,
            root,
            dnn,
            input_bounds,
        }
    }

    fn expand_node(&mut self, id: NodeId) -> Vec<NodeId> {
        let node = self.arena.get_mut(id).unwrap().get_mut();
        let children = node.expand(&self.dnn);
        children
            .into_iter()
            .map(|x| self.arena.new_node(x))
            .collect()
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

    /// # Panics
    #[allow(clippy::too_many_lines)]
    pub fn sample_safe_star(
        &mut self,
        loc: &Array1<T>,
        scale: &Array2<T>,
        safe_value: T,
        cdf_samples: usize,
        max_iters: usize,
    ) -> Option<(StarNode<T>, T)> {
        let mut rng = rand::thread_rng();
        let mut current_node = self.root;
        let mut path = vec![];
        let mut path_logp = T::zero();
        let root = self.root;
        let infeasible_reset = |arena: &mut Arena<StarNode<T>>,
                                x: NodeId,
                                path: &mut Vec<NodeId>,
                                path_logp: &mut T|
         -> NodeId {
            arena[x].get_mut().set_feasible(false);
            let mut infeas_cdf =
                arena[x]
                    .get_mut()
                    .gaussian_cdf(loc, scale, cdf_samples, max_iters);
            path.drain(..).rev().for_each(|x| {
                // check if all chilren are infeasible
                if x.children(arena).any(|x| arena[x].get().is_feasible) {
                    // if not infeasible, update CDF
                    arena[x].get_mut().add_cdf(T::neg(T::one()) * infeas_cdf);
                } else {
                    arena[x].get_mut().set_feasible(false);
                    infeas_cdf = arena[x]
                        .get_mut()
                        .gaussian_cdf(loc, scale, cdf_samples, max_iters)
                }
            });
            *path_logp = T::zero();
            root
        };
        loop {
            // base case for feasability
            if current_node == self.root && !self.arena[current_node].get().get_feasible() {
                return None;
            }
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds = self.arena[current_node]
                    .get_mut()
                    .get_output_bounds(&self.dnn, 0);
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
                let children: Vec<NodeId> = if self.arena[current_node].get().get_expanded() {
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

#[derive(Debug, Clone)]
pub struct StarNode<T: num::Float> {
    star: Star<T, IxDyn>,
    dnn_layer: usize,
    remaining_steps: Option<usize>,
    cdf: Option<T>,
    output_bounds: Option<(T, T)>,
    is_feasible: bool,
    is_expanded: bool,
}

impl<T: num::Float> StarNode<T>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign,
    f64: std::convert::From<T>,
{
    pub fn get_feasible(&self) -> bool {
        self.is_feasible
    }

    pub fn set_feasible(&mut self, val: bool) {
        self.is_feasible = val
    }

    pub fn gaussian_cdf(
        &mut self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
    ) -> T {
        self.cdf.map_or_else(
            || {
                let out = self.star.trunc_gaussian_cdf(mu, sigma, n, max_iters);
                let cdf = out.0.into();
                self.cdf = Some(cdf);
                cdf
            },
            |cdf| cdf,
        )
    }

    pub fn set_cdf(&mut self, val: T) {
        self.cdf = Some(val);
    }

    /// # Panics
    pub fn add_cdf(&mut self, add: T) {
        if let Some(ref mut cdf) = self.cdf {
            *cdf += add
        } else {
            todo!()
        }
    }

    pub fn gaussian_sample<R: Rng>(
        &self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        input_bounds: &Option<Bounds1<T>>,
    ) -> Vec<(Array1<T>, T)> {
        self.star
            .clone()
            .gaussian_sample(rng, mu, sigma, n, max_iters, input_bounds)
    }

    pub fn get_output_bounds(&mut self, dnn: &DNN<T>, idx: usize) -> (T, T) {
        self.output_bounds.map_or_else(
            || {
                // TODO: update this to use DeepPoly to get proper bounds rather than this estimate
                let bounds = {
                    let out_star = dnn
                        .get_layers()
                        .iter()
                        .skip(self.dnn_layer)
                        .fold(self.star.clone(), |s: Star<T, IxDyn>, l: &Layer<T>| {
                            l.apply_star(s)
                        });
                    (out_star.clone().get_min(idx), out_star.get_max(idx))
                };
                self.output_bounds = Some(bounds);
                bounds
            },
            |bounds| bounds,
        )
    }

    pub fn get_expanded(&self) -> bool {
        self.is_expanded
    }

    pub fn set_expanded(&mut self) {
        self.is_expanded = true;
    }

    /// # Panics
    pub fn expand(&self, dnn: &DNN<T>) -> Vec<Self> {
        // check if there is a step relu to do
        if let Some(relu_step) = self.remaining_steps {
            let new_child_stars = self.star.clone().step_relu(relu_step);
            let new_remaining_steps = if relu_step == 0 {
                None
            } else {
                Some(relu_step - 1)
            };
            new_child_stars
                .into_iter()
                .map(|star| Self {
                    star,
                    dnn_layer: self.dnn_layer,
                    remaining_steps: new_remaining_steps,
                    cdf: None,
                    output_bounds: None,
                    is_expanded: false,
                    is_feasible: true,
                })
                .collect()
        } else if let Some(layer) = dnn.get_layer(self.dnn_layer) {
            vec![Self {
                star: layer.apply_star(self.star.clone()),
                dnn_layer: self.dnn_layer + 1,
                remaining_steps: Some(
                    dnn.get_layer(self.dnn_layer).unwrap().output_shape()[-1].unwrap() - 1,
                ),
                cdf: None,
                output_bounds: None,
                is_expanded: false,
                is_feasible: true,
            }]
        } else {
            // leaf node
            vec![]
        }
    }
}

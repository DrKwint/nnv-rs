#![allow(non_snake_case)]
use crate::dnn::DNNIterator;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeOp;
use crate::star_node::StarNodeType;
use crate::util::ArenaLike;
use crate::Bounds;
use crate::NNVFloat;
use crate::DNN;
use log::debug;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use num::Float;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use std::iter::Sum;

/// Data structure representing the paths through a deep neural network (DNN)
#[derive(Debug)]
pub struct Constellation<T: NNVFloat, D: Dimension> {
    arena: Vec<StarNode<T, D>>,
    children: Vec<Option<StarNodeType<T>>>,
    cdf: Vec<Option<T>>,
    loc: Array1<T>,
    scale: Array2<T>,
    dnn: DNN<T>,
    input_bounds: Option<Bounds<T, D>>,
}

impl<T: NNVFloat, D: Dimension> Constellation<T, D>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::AddAssign,
    f64: std::convert::From<T>,
{
    /// Instantiate a Constellation with given input set and network
    pub fn new(
        input_star: Star<T, D>,
        dnn: DNN<T>,
        input_bounds: Option<Bounds<T, D>>,
        loc: Array1<T>,
        scale: Array2<T>,
    ) -> Self {
        let star_node = StarNode::default(input_star);
        let children = vec![None];
        let arena = vec![star_node];
        let cdf = vec![];
        Self {
            arena,
            children,
            cdf,
            loc,
            scale,
            dnn,
            input_bounds,
        }
    }

    pub fn get_dnn(&self) -> &DNN<T> {
        &self.dnn
    }

    pub fn get_input_bounds(&self) -> &Option<Bounds<T, D>> {
        &self.input_bounds
    }

    fn add_node(&mut self, node: StarNode<T, D>) -> usize {
        let child_idx = self.arena.new_node(node);
        let other_child_idx = self.children.new_node(None);
        debug_assert_eq!(child_idx, other_child_idx);
        child_idx
    }

    pub fn reset_input_distribution(&mut self, loc: Array1<T>, scale: Array2<T>) {
        self.loc = loc;
        self.scale = scale;
        self.cdf.clear();
    }

    pub fn reset_with_star(&mut self, input_star: Star<T, D>, input_bounds: Option<Bounds<T, D>>) {
        let star_node = StarNode::default(input_star);
        self.arena = vec![star_node];
        self.children = vec![None];
        self.input_bounds = input_bounds;
    }

    pub fn get_child_ids(&self, node_id: usize) -> Option<Vec<usize>> {
        match self.children[node_id] {
            Some(StarNodeType::Leaf) => Some(vec![]),
            Some(StarNodeType::Affine { child_idx }) => Some(vec![child_idx]),
            Some(StarNodeType::StepRelu {
                dim: _,
                fst_child_idx,
                snd_child_idx,
            }) => {
                let mut child_ids: Vec<usize> = vec![fst_child_idx];
                if let Some(idx) = snd_child_idx {
                    child_ids.push(idx);
                }
                Some(child_ids)
            }
            Some(StarNodeType::StepReluDropOut {
                dim,
                dropout_prob,
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
            }) => {
                let mut child_ids: Vec<usize> = vec![fst_child_idx];
                if let Some(idx) = snd_child_idx {
                    child_ids.push(idx);
                }
                if let Some(idx) = trd_child_idx {
                    child_ids.push(idx);
                }
                Some(child_ids)
            }
            None => None,
        }
    }
}

impl<T: NNVFloat> Constellation<T, Ix2>
where
    f64: std::convert::From<T>,
{
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
        safe_value: T,
        cdf_samples: usize,
        num_samples: usize,
        max_iters: usize,
    ) -> (Vec<(Array1<T>, T)>, T) {
        let input_bounds = self.input_bounds.clone();
        if let Some((safe_star, path_logp)) =
            self.sample_safe_star(safe_value, cdf_samples, max_iters)
        {
            (
                safe_star.gaussian_sample(
                    rng,
                    &self.loc,
                    &self.scale,
                    num_samples,
                    max_iters,
                    &input_bounds,
                ),
                path_logp,
            )
        } else {
            panic!()
        }
    }
}

impl<T: crate::NNVFloat> Constellation<T, Ix2>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::AddAssign
        + Default
        + Sum,
    f64: std::convert::From<T>,
{
    ///
    /// # Panics
    ///
    /// loc, scale: Input Gaussian
    /// safe_value
    #[allow(clippy::too_many_lines)]
    pub fn sample_safe_star(
        &mut self,
        safe_value: T,
        cdf_samples: usize,
        max_iters: usize,
    ) -> Option<(StarNode<T, Ix2>, T)> {
        debug!("Running sample_safe_star with: safe_value {}", safe_value);
        let mut rng = rand::thread_rng();
        let mut current_node = 0;
        let mut path = vec![];
        let mut path_logp = T::zero();
        let input_bounds = self.input_bounds.clone();
        let infeasible_reset = |me: &mut Self,
                                x: usize,
                                path: &mut Vec<usize>,
                                path_logp: &mut T|
         -> usize {
            debug!("Infeasible reset!");
            me.arena[x].set_feasible(false);
            let mut infeas_cdf =
                me.arena[x].gaussian_cdf(&me.loc, &me.scale, cdf_samples, max_iters, &input_bounds);
            path.drain(..).rev().for_each(|x| {
                // check if all chilren are infeasible
                if me
                    .get_child_ids(x)
                    .unwrap()
                    .iter()
                    .any(|x| me.arena[*x].get_feasible())
                {
                    // if not infeasible, update CDF
                    me.arena[x].add_cdf(T::neg(T::one()) * infeas_cdf);
                } else {
                    me.arena[x].set_feasible(false);
                    infeas_cdf = me.arena[x].gaussian_cdf(
                        &me.loc,
                        &me.scale,
                        cdf_samples,
                        max_iters,
                        &input_bounds,
                    )
                }
            });
            *path_logp = T::zero();
            0
        };
        loop {
            debug!("Current node: {:?}", current_node);
            // base case for feasability
            if current_node == 0 && !self.arena[current_node].get_feasible() {
                return None;
            }
            // check feasibility of current node
            {
                // makes the assumption that bounds are on 0th dimension of output
                let output_bounds = self.arena[current_node]
                    .get_output_bounds(&self.dnn, &|x| (x.lower()[[0]], x.upper()[[0]]));
                debug!("Output bounds: {:?}", output_bounds);
                if output_bounds.1 <= safe_value {
                    // handle case where star is safe
                    let safe_star = self.arena[current_node].clone();
                    return Some((safe_star, path_logp));
                } else if output_bounds.0 > safe_value {
                    // handle case where star is infeasible
                    self.arena[current_node].set_feasible(false);
                    current_node = infeasible_reset(self, current_node, &mut path, &mut path_logp);
                    continue;
                } else if let StarNodeType::Leaf = self.get_children(current_node) {
                    // do procedure to select safe part
                    let safe_star = self.arena[current_node].get_safe_star(safe_value);
                    return Some((safe_star, path_logp));
                }
                // otherwise, push to path and continue expanding
                path.push(current_node);
            }
            // expand node
            {
                // if let StarNodeType::Leaf =
                //     self.arena[current_node].get_children(&mut self.arena, &self.dnn)
                // {
                //     let safe_star = self.arena[current_node].clone();
                //     return Some((safe_star, path_logp));
                // }
                let result =
                    self.select_child(current_node, path_logp, cdf_samples, max_iters, &mut rng);
                match result {
                    Some((node, logp)) => {
                        current_node = node;
                        path_logp += logp;
                    }
                    None => {
                        current_node =
                            infeasible_reset(self, current_node, &mut path, &mut path_logp);
                    }
                }
            }
        }
    }

    /// Returns the children of a node
    ///
    /// Lazily loads children into the arena and returns a reference to them.
    ///
    /// # Arguments
    ///
    /// * `self` - The node to expand
    /// * `node_arena` - The data structure storing star nodes
    /// * `dnn_iter` - The iterator of operations in the dnn
    ///
    /// # Returns
    /// * `children` - `StarNodeType<T>`
    ///
    /// # Panics
    pub fn get_children(&mut self, node_id: usize) -> &StarNodeType<T> {
        if self
            .children
            .get(node_id)
            .and_then(std::option::Option::as_ref)
            .is_some()
        {
            self.children
                .get(node_id)
                .and_then(std::option::Option::as_ref)
                .unwrap()
        } else {
            self.expand(node_id)
        }
    }

    /// Expand a node's children, inserting them into the arena.
    ///
    /// # Arguments
    ///
    /// * `self` - The node to expand
    /// * `node_arena` - The data structure storing star nodes
    /// * `dnn_iter` - The iterator of operations in the dnn
    ///
    /// # Returns
    /// * `children` - `StarNodeType<T>`
    fn expand(&mut self, node_id: usize) -> &StarNodeType<T> {
        let dnn_index = self.arena[node_id].get_dnn_index();
        let dnn_iter = &mut DNNIterator::new(&self.dnn, dnn_index);

        // Get this node's operation from the dnn_iter
        let op = dnn_iter.next();
        // Do this node's operation to produce its children
        let children = match op {
            Some(StarNodeOp::Leaf) => StarNodeType::Leaf,
            Some(StarNodeOp::Affine(aff)) => {
                let child_idx = self.add_node(
                    StarNode::default(self.arena[node_id].get_star().affine_map2(&aff))
                        .with_dnn_index(dnn_iter.get_idx()),
                );
                StarNodeType::Affine { child_idx }
            }
            Some(StarNodeOp::StepRelu(dim)) => {
                let child_stars = self.arena[node_id].get_star().step_relu2(dim);
                let dnn_idx = dnn_iter.get_idx();
                let ids: Vec<usize> = child_stars
                    .into_iter()
                    .map(|star| self.add_node(StarNode::default(star).with_dnn_index(dnn_idx)))
                    .collect();
                StarNodeType::StepRelu {
                    dim,
                    fst_child_idx: ids[0],
                    snd_child_idx: ids.get(1).copied(),
                }
            }
            Some(StarNodeOp::StepReluDropout((dropout_prob, dim))) => {
                let child_stars = self.arena[node_id].get_star().step_relu2_dropout(dim);
                let dnn_idx = dnn_iter.get_idx();
                let ids: Vec<usize> = child_stars
                    .into_iter()
                    .map(|star| self.add_node(StarNode::default(star).with_dnn_index(dnn_idx)))
                    .collect();
                StarNodeType::StepReluDropOut {
                    dim,
                    dropout_prob,
                    fst_child_idx: ids[0],
                    snd_child_idx: ids.get(1).copied(),
                    trd_child_idx: ids.get(2).copied(),
                }
            }
            None => panic!(),
        };
        self.children[node_id] = Some(children);
        self.children
            .get(node_id)
            .and_then(std::option::Option::as_ref)
            .unwrap()
    }

    fn select_child(
        &mut self,
        current_node: usize,
        mut path_logp: T,
        cdf_samples: usize,
        max_iters: usize,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Option<(usize, T)> {
        let input_bounds = self.input_bounds.clone();
        match *self.get_children(current_node) {
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
                    .filter(|child| self.arena[*child].get_feasible())
                    .collect();
                match children.len() {
                    0 => None,
                    1 => {
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
                        Some((children[0], path_logp))
                    }
                    2 => {
                        let fst_cdf = self.arena[children[0]].gaussian_cdf(
                            &self.loc,
                            &self.scale,
                            cdf_samples,
                            max_iters,
                            &input_bounds,
                        );
                        let snd_cdf = self.arena[children[1]].gaussian_cdf(
                            &self.loc,
                            &self.scale,
                            cdf_samples,
                            max_iters,
                            &input_bounds,
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
                                let parent_cdf = self.arena[current_node].gaussian_cdf(
                                    &self.loc,
                                    &self.scale,
                                    cdf_samples,
                                    max_iters,
                                    &input_bounds,
                                );
                                let mut derived_fst_cdf = parent_cdf - snd_cdf;
                                if derived_fst_cdf.is_sign_negative() {
                                    derived_fst_cdf = T::epsilon();
                                }
                                self.arena[children[0]].set_cdf(derived_fst_cdf);
                                derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                            }
                            (true, false) => {
                                let parent_cdf = self.arena[current_node].gaussian_cdf(
                                    &self.loc,
                                    &self.scale,
                                    cdf_samples,
                                    max_iters,
                                    &input_bounds,
                                );
                                let mut derived_snd_cdf = parent_cdf - fst_cdf;
                                if derived_snd_cdf.is_sign_negative() {
                                    derived_snd_cdf = T::epsilon();
                                }
                                self.arena[children[1]].set_cdf(derived_snd_cdf);
                                fst_cdf / (fst_cdf + derived_snd_cdf)
                            }
                            // If both CDFs are non-normal, we'll mark the current node as infeasible
                            (false, false) => {
                                return None;
                            }
                        };
                        // Safe unwrap due to above handling
                        let dist = Bernoulli::new(fst_prob.clone().into()).expect(&format!("Bad fst_prob: {:?} fst_cdf: {:?} snd_cdf: {:?}, fst_n {:?}, snd_n {:?}", fst_prob, fst_cdf, snd_cdf, fst_cdf.is_normal(), snd_cdf.is_normal()));
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
                        .filter(|child| self.arena[*child].get_feasible())
                        .collect();
                    match children.len() {
                        0 => None,
                        1 => Some((children[0], path_logp)),
                        _ => {
                            let fst_cdf = self.arena[children[0]].gaussian_cdf(
                                &self.loc,
                                &self.scale,
                                cdf_samples,
                                max_iters,
                                &input_bounds,
                            );
                            let snd_cdf = self.arena[children[1]].gaussian_cdf(
                                &self.loc,
                                &self.scale,
                                cdf_samples,
                                max_iters,
                                &input_bounds,
                            );

                            // Handle the case where a CDF gives a non-normal value
                            let fst_prob = match (fst_cdf.is_normal(), snd_cdf.is_normal()) {
                                (true, true) => fst_cdf / (fst_cdf + snd_cdf),
                                (false, true) => {
                                    let parent_cdf = self.arena[current_node].gaussian_cdf(
                                        &self.loc,
                                        &self.scale,
                                        cdf_samples,
                                        max_iters,
                                        &input_bounds,
                                    );
                                    let derived_fst_cdf = parent_cdf - snd_cdf;
                                    self.arena[children[0]].set_cdf(derived_fst_cdf);
                                    derived_fst_cdf / (derived_fst_cdf + snd_cdf)
                                }
                                (true, false) => {
                                    let parent_cdf = self.arena[current_node].gaussian_cdf(
                                        &self.loc,
                                        &self.scale,
                                        cdf_samples,
                                        max_iters,
                                        &input_bounds,
                                    );
                                    let derived_snd_cdf = parent_cdf - fst_cdf;
                                    self.arena[children[1]].set_cdf(derived_snd_cdf);
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
mod tests {
    use crate::affine::Affine2;
    use crate::test_util::*;
    use ndarray::Array;
    use ndarray::Array2;
    use ndarray::ArrayView;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_sample_safe_star(mut constellation in generic_constellation(2, 2, 2, 2)) {
            constellation.sample_safe_star(1.0, 1, 1);
        }
    }
}

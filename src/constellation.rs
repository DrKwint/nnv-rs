#![allow(non_snake_case)]
use crate::bounds::Bounds1;
use crate::dnn::DNNIterator;
use crate::dnn::{DNNIndex, DNN};
use crate::gaussian::GaussianDistribution;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeOp;
use crate::star_node::StarNodeType;
use crate::util::ArenaLike;
use crate::NNVFloat;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Data structure representing the paths through a deep neural network (DNN)
#[derive(Debug)]
pub struct Constellation<T: NNVFloat, D: Dimension> {
    arena: Vec<StarNode<T, D>>,
    node_type: Vec<Option<StarNodeType<T>>>,
    parents: Vec<Option<usize>>,
    loc: Array1<T>,
    scale: Array2<T>,
    dnn: DNN<T>,
}

impl<T: NNVFloat, D: Dimension> Constellation<T, D> {
    /// Instantiate a Constellation with given input set and network
    pub fn new(input_star: Star<T, D>, dnn: DNN<T>, loc: Array1<T>, scale: Array2<T>) -> Self {
        let star_node = StarNode::default(input_star, None);
        let node_type = vec![None];
        let arena = vec![star_node];
        let parents = vec![None];
        Self {
            arena,
            node_type,
            parents,
            loc,
            scale,
            dnn,
        }
    }

    pub fn get_dnn(&self) -> &DNN<T> {
        &self.dnn
    }

    pub fn get_root_id(&self) -> usize {
        0
    }

    pub fn get_loc(&self) -> &Array1<T> {
        &self.loc
    }

    pub fn get_scale(&self) -> &Array2<T> {
        &self.scale
    }

    pub fn reset_input_distribution(&mut self, loc: Array1<T>, scale: Array2<T>) {
        self.loc = loc;
        self.scale = scale;
        self.arena.iter_mut().for_each(StarNode::reset_cdf);
    }

    pub fn reset_with_star(&mut self, input_star: Star<T, D>) {
        let star_node = StarNode::default(input_star, None);
        self.arena = vec![star_node];
        self.node_type = vec![None];
    }
}

impl<T: crate::NNVFloat> Constellation<T, Ix2> {
    pub fn get_input_bounds(&self) -> Option<&Bounds1<T>> {
        self.arena[0].get_input_bounds()
    }

    fn add_node_with_gauss_init(
        &mut self,
        node: StarNode<T, Ix2>,
        parent_id: usize,
        stability_eps: T,
    ) -> usize {
        let child_idx = self.arena.new_node(node);
        let other_child_idx = self.node_type.new_node(None);
        let other_other_child_idx = self.parents.new_node(Some(parent_id));
        debug_assert_eq!(child_idx, other_child_idx);
        debug_assert_eq!(child_idx, other_other_child_idx);
        debug_assert!(self.try_get_gaussian_distribution(parent_id).is_some());
        self.initialize_node_tilting_from_parent(child_idx, parent_id, 20, stability_eps);
        child_idx
    }

    /// # Panics
    pub fn initialize_node_tilting_from_parent(
        &mut self,
        node_id: usize,
        parent_id: usize,
        max_accept_reject_iters: usize,
        stability_eps: T,
    ) {
        let parent_tilting_soln = {
            debug_assert!(self.try_get_gaussian_distribution(parent_id).is_some());
            let parent_distr = self.arena[parent_id]
                .try_get_gaussian_distribution()
                .unwrap();
            // This isn't always the case because the parent may be unconstrained (if it's the root)
            //debug_assert!(parent_distr.try_get_tilting_solution().is_some());
            parent_distr.try_get_tilting_solution().cloned()
        };
        let child_distr = self.arena[node_id].get_gaussian_distribution(
            &self.loc,
            &self.scale,
            max_accept_reject_iters,
            stability_eps,
        );
        child_distr.populate_tilting_solution(parent_tilting_soln.as_ref());
    }

    pub fn try_get_gaussian_distribution(
        &self,
        node_id: usize,
    ) -> Option<&GaussianDistribution<T>> {
        self.arena[node_id].try_get_gaussian_distribution()
    }

    pub fn get_node_input_bounds(&self, node_id: usize) -> Option<&Bounds1<T>> {
        self.arena[node_id].get_input_bounds()
    }

    pub fn get_node_dnn_index(&self, node_id: usize) -> DNNIndex {
        self.arena[node_id].get_dnn_index()
    }

    pub fn get_node_output_bounds(&mut self, node_id: usize) -> (T, T) {
        self.arena[node_id].get_output_bounds(&self.dnn, &|x| (x.lower()[[0]], x.upper()[[0]]))
    }

    pub fn is_node_leaf(&mut self, id: usize, stability_eps: T) -> bool {
        *self.get_node_type(id, stability_eps) == StarNodeType::Leaf
    }

    pub fn get_node_cdf<R: Rng>(
        &mut self,
        node_id: usize,
        cdf_samples: usize,
        max_iters: usize,
        rng: &mut R,
        stability_eps: T,
    ) -> T {
        self.arena[node_id].gaussian_cdf(
            &self.loc,
            &self.scale,
            cdf_samples,
            max_iters,
            rng,
            stability_eps,
        )
    }

    pub fn set_node_cdf(&mut self, node_id: usize, cdf: T) {
        self.arena[node_id].set_cdf(cdf);
    }

    pub fn add_node_cdf(&mut self, node_id: usize, cdf: T) {
        self.arena[node_id].add_cdf(cdf);
    }

    pub fn sample_gaussian_node_output<R: Rng>(
        &mut self,
        node_id: usize,
        rng: &mut R,
        n: usize,
        max_iters: usize,
        stability_eps: T,
    ) -> Vec<Array1<T>> {
        let node = &mut self.arena[node_id];
        node.gaussian_sample(
            rng,
            &self.loc,
            &self.scale,
            n,
            max_iters,
            &None,
            stability_eps,
        )
        .into_iter()
        .map(|input| node.forward(&input))
        .collect()
    }

    /// # Panics
    pub fn sample_gaussian_node<R: Rng>(
        &mut self,
        node_id: usize,
        rng: &mut R,
        n: usize,
        max_iters: usize,
        stability_eps: T,
    ) -> Vec<Array1<T>> {
        let initialization_opt = self.parents[node_id].and_then(|x| {
            self.arena[x]
                .try_get_gaussian_distribution()
                .unwrap()
                .try_get_tilting_solution()
                .cloned()
        });
        self.arena[node_id].gaussian_sample(
            rng,
            &self.loc,
            &self.scale,
            n,
            max_iters,
            &initialization_opt,
            stability_eps,
        )
    }

    /// # Panics
    pub fn sample_gaussian_node_safe<R: Rng>(
        &mut self,
        node_id: usize,
        rng: &mut R,
        n: usize,
        max_iters: usize,
        safe_value: T,
        stability_eps: T,
    ) -> Vec<Array1<T>> {
        let mut safe_star = self.arena[node_id].get_safe_star(safe_value);
        let initialization_opt = self.parents[node_id].map(|x| {
            self.arena[x]
                .try_get_gaussian_distribution()
                .unwrap()
                .try_get_tilting_solution()
                .unwrap()
                .clone()
        });
        safe_star.gaussian_sample(
            rng,
            &self.loc,
            &self.scale,
            n,
            max_iters,
            &initialization_opt,
            stability_eps,
        )
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
    pub fn get_node_type(&mut self, node_id: usize, stability_eps: T) -> &StarNodeType<T> {
        if self
            .node_type
            .get(node_id)
            .and_then(std::option::Option::as_ref)
            .is_some()
        {
            self.node_type
                .get(node_id)
                .and_then(std::option::Option::as_ref)
                .unwrap()
        } else {
            debug_assert!(self.try_get_gaussian_distribution(node_id).is_some());
            self.expand(node_id, stability_eps)
        }
    }

    pub fn get_node_child_ids(&mut self, node_id: usize, stability_eps: T) -> Vec<usize> {
        match self.get_node_type(node_id, stability_eps) {
            StarNodeType::Leaf => vec![],
            StarNodeType::Affine { child_idx } => vec![*child_idx],
            StarNodeType::StepRelu {
                dim: _,
                fst_child_idx,
                snd_child_idx,
            } => {
                let mut child_ids: Vec<usize> = vec![*fst_child_idx];
                if let Some(idx) = snd_child_idx {
                    child_ids.push(*idx);
                }
                child_ids
            }
            StarNodeType::StepReluDropOut {
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
                ..
            } => {
                let mut child_ids: Vec<usize> = vec![*fst_child_idx];
                if let Some(idx) = snd_child_idx {
                    child_ids.push(*idx);
                }
                if let Some(idx) = trd_child_idx {
                    child_ids.push(*idx);
                }
                child_ids
            }
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
    fn expand(&mut self, node_id: usize, stability_eps: T) -> &StarNodeType<T> {
        let dnn_index = self.arena[node_id].get_dnn_index();
        let dnn_iter = &mut DNNIterator::new(&self.dnn, dnn_index);

        // Get this node's operation from the dnn_iter
        let op = dnn_iter.next();
        debug_assert!(self.arena[node_id]
            .try_get_gaussian_distribution()
            .is_some());
        // Do this node's operation to produce its children
        let children = match op {
            Some(StarNodeOp::Leaf) => StarNodeType::Leaf,
            Some(StarNodeOp::Affine(aff)) => {
                let child_idx = self.add_node_with_gauss_init(
                    StarNode::default(self.arena[node_id].get_star().affine_map2(&aff), None)
                        .with_dnn_index(dnn_iter.get_idx()),
                    node_id,
                    stability_eps,
                );
                StarNodeType::Affine { child_idx }
            }
            Some(StarNodeOp::StepRelu(dim)) => {
                let child_stars = self.arena[node_id].get_star().step_relu2(dim);
                let dnn_idx = dnn_iter.get_idx();

                let mut ids = vec![];

                if let Some(lower_star) = child_stars.0 {
                    let mut bounds = self.arena[node_id].calculate_star_local_bounds().clone();
                    bounds.index_mut(dim)[0] = T::zero();
                    bounds.index_mut(dim)[1] = T::zero();
                    let lower_node =
                        StarNode::default(lower_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node_with_gauss_init(lower_node, node_id, stability_eps);
                    ids.push(id);
                }

                if let Some(upper_star) = child_stars.1 {
                    let mut bounds = self.arena[node_id].calculate_star_local_bounds().clone();
                    let mut lb = bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = T::zero();
                    }
                    let upper_node =
                        StarNode::default(upper_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node_with_gauss_init(upper_node, node_id, stability_eps);
                    ids.push(id);
                }

                StarNodeType::StepRelu {
                    dim,
                    fst_child_idx: ids[0],
                    snd_child_idx: ids.get(1).copied(),
                }
            }
            Some(StarNodeOp::StepReluDropout((dropout_prob, dim))) => {
                let child_stars = self.arena[node_id].get_star().step_relu2_dropout(dim);
                let dnn_idx = dnn_iter.get_idx();
                let mut ids = vec![];

                if let Some(dropout_star) = child_stars.0 {
                    let mut bounds = self.arena[node_id].calculate_star_local_bounds().clone();
                    bounds.index_mut(dim)[0] = T::zero();
                    bounds.index_mut(dim)[1] = T::zero();
                    let dropout_node =
                        StarNode::default(dropout_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node_with_gauss_init(dropout_node, node_id, stability_eps);
                    ids.push(id);
                }

                if let Some(lower_star) = child_stars.1 {
                    let mut bounds = self.arena[node_id].calculate_star_local_bounds().clone();
                    bounds.index_mut(dim)[0] = T::zero();
                    bounds.index_mut(dim)[1] = T::zero();
                    let lower_node =
                        StarNode::default(lower_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node_with_gauss_init(lower_node, node_id, stability_eps);
                    ids.push(id);
                }

                if let Some(upper_star) = child_stars.2 {
                    let mut bounds = self.arena[node_id].calculate_star_local_bounds().clone();
                    let mut lb = bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = T::zero();
                    }
                    let upper_node =
                        StarNode::default(upper_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node_with_gauss_init(upper_node, node_id, stability_eps);
                    ids.push(id);
                }

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
        self.node_type[node_id] = Some(children);
        self.node_type
            .get(node_id)
            .and_then(std::option::Option::as_ref)
            .unwrap()
    }
}

#[cfg(test)]
mod test {
    use crate::asterism::Asterism;
    use crate::test_util::*;
    use proptest::*;

    proptest! {
        #[test]
        fn test_cache_bounds_equivalence(mut constellation in generic_constellation(2, 2, 2, 2)) {
            let mut rng = rand::thread_rng();
            let mut asterism = Asterism::new(&mut constellation, 1.);
            asterism.sample_safe_star(1, &mut rng, 1, 1, None, 1e-4);

            for (i, node) in constellation.arena.iter_mut().enumerate() {
                if let Some(local_bounds) = node.calculate_star_local_bounds_direct() {
                    let bounds = local_bounds.clone();
                    node.set_local_bounds_direct(None);
                    let expected_local_bounds = node.calculate_star_local_bounds();

                    prop_assert!(bounds.bounds_iter().into_iter()
                        .zip(expected_local_bounds.bounds_iter().into_iter())
                        .all(|(b1, b2)| b1.abs_diff_eq(&b2, 1e-8)),
                        "\nBounds: {:?}\nExpected: {:?}", &bounds, expected_local_bounds);
                }
            }
        }
    }
}

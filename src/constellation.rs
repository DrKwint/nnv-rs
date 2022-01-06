use crate::dnn::{DNNIndex, DNN};
use crate::probstarset::{ProbStarSet, ProbStarSet2};
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeType;
use crate::starset::{StarSet, StarSet2};
use crate::util::ArenaLike;
use crate::NNVFloat;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constellation<D: Dimension> {
    arena: Vec<StarNode<D>>,
    node_type: Vec<Option<StarNodeType>>,
    parents: Vec<Option<usize>>,
    loc: Array1<NNVFloat>,
    scale: Array2<NNVFloat>,
    dnn: DNN,
    max_accept_reject_iters: usize,
    num_cdf_samples: usize,
    stability_eps: NNVFloat,
}

impl<D: Dimension> Constellation<D> {
    pub fn new(
        dnn: DNN,
        input_star: Star<D>,
        loc: Array1<NNVFloat>,
        scale: Array2<NNVFloat>,
        max_accept_reject_iters: usize,
        num_cdf_samples: usize,
        stability_eps: NNVFloat,
    ) -> Self {
        let star_node = StarNode::default(input_star, None);
        let arena = vec![star_node];
        let node_type = vec![None];
        let parents = vec![None];
        Self {
            arena,
            node_type,
            parents,
            loc,
            scale,
            dnn,
            max_accept_reject_iters,
            num_cdf_samples,
            stability_eps,
        }
    }
}

impl<D: Dimension> StarSet<D> for Constellation<D> {
    fn get_node(&self, node_id: usize) -> &StarNode<D> {
        &self.arena[node_id]
    }

    fn get_node_mut(&mut self, node_id: usize) -> &mut StarNode<D> {
        &mut self.arena[node_id]
    }

    fn add_node(&mut self, node: StarNode<D>, parent_id: usize) -> usize {
        let child_idx = self.arena.new_node(node);
        let other_child_idx = self.node_type.new_node(None);
        let other_other_child_idx = self.parents.new_node(Some(parent_id));
        debug_assert_eq!(child_idx, other_child_idx);
        debug_assert_eq!(child_idx, other_other_child_idx);
        child_idx
    }

    fn get_dnn(&self) -> &DNN {
        &self.dnn
    }

    fn try_get_node_parent_id(&self, node_id: usize) -> Option<usize> {
        self.parents[node_id]
    }

    fn get_node_dnn_index(&self, node_id: usize) -> DNNIndex {
        self.arena[node_id].get_dnn_index()
    }

    fn try_get_node_type(&self, node_id: usize) -> &Option<StarNodeType> {
        &self.node_type[node_id]
    }

    fn set_node_type(&mut self, node_id: usize, children: StarNodeType) {
        self.node_type[node_id] = Some(children);
    }

    fn reset_with_star(&mut self, input_star: Star<D>) {
        let star_node = StarNode::default(input_star, None);
        self.arena = vec![star_node];
        self.node_type = vec![None];
        self.parents = vec![None];
    }
}

impl StarSet2 for Constellation<Ix2> {
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
    fn get_node_type(&mut self, node_id: usize) -> &StarNodeType {
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
            self.expand(node_id)
        }
    }
}

impl<D: Dimension> ProbStarSet<D> for Constellation<D> {
    fn reset_input_distribution(&mut self, loc: Array1<NNVFloat>, scale: Array2<NNVFloat>) {
        self.loc = loc;
        self.scale = scale;
        self.arena.iter_mut().for_each(StarNode::reset_cdf);
    }
}

impl ProbStarSet2 for Constellation<Ix2> {
    fn get_node_mut_with_borrows(
        &mut self,
        node_id: usize,
    ) -> (
        &mut StarNode<Ix2>,
        ArrayView1<NNVFloat>,
        ArrayView2<NNVFloat>,
        &DNN,
    ) {
        (
            &mut self.arena[node_id],
            self.loc.view(),
            self.scale.view(),
            &self.dnn,
        )
    }

    fn get_loc(&self) -> ArrayView1<NNVFloat> {
        self.loc.view()
    }

    fn get_scale(&self) -> ArrayView2<NNVFloat> {
        self.scale.view()
    }

    fn get_max_accept_reject_iters(&self) -> usize {
        self.max_accept_reject_iters
    }

    fn get_stability_eps(&self) -> NNVFloat {
        self.stability_eps
    }

    fn get_cdf_samples(&self) -> usize {
        self.num_cdf_samples
    }
}

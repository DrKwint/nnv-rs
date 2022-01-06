use crate::dnn::DNNIndex;
use crate::dnn::DNN;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeType;
use crate::starset::StarSet;
use crate::starset::StarSet2;
use crate::util::ArenaLike;
use ndarray::Dimension;
use ndarray::Ix2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecStarSet<D: Dimension> {
    arena: Vec<StarNode<D>>,
    node_type: Vec<Option<StarNodeType>>,
    parents: Vec<Option<usize>>,
    dnn: DNN,
}

impl<D: Dimension> VecStarSet<D> {
    pub fn new(dnn: DNN, input_star: Star<D>) -> Self {
        let star_node = StarNode::default(input_star, None);
        let arena = vec![star_node];
        let node_type = vec![None];
        let parents = vec![None];
        Self {
            arena,
            node_type,
            parents,
            dnn,
        }
    }
}

impl<D: Dimension> StarSet<D> for VecStarSet<D> {
    fn get_node(&self, node_id: usize) -> &StarNode<D> {
        &self.arena[node_id]
    }

    fn get_node_mut(&mut self, node_id: usize) -> &mut StarNode<D> {
        todo!()
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
        todo!()
    }

    fn get_node_dnn_index(&self, node_id: usize) -> DNNIndex {
        self.arena[node_id].get_dnn_index()
    }

    fn try_get_node_type(&self, node_id: usize) -> &Option<StarNodeType> {
        &self.node_type[node_id]
    }

    fn set_node_type(&mut self, node_id: usize, children: StarNodeType) {
        todo!()
    }

    fn reset_with_star(&mut self, input_star: Star<D>) {
        todo!()
    }
}

impl StarSet2 for VecStarSet<Ix2> {
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

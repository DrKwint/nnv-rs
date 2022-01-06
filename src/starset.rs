use crate::bounds::Bounds1;
use crate::dnn::DNNIndex;
use crate::dnn::DNNIterator;
use crate::dnn::DNN;
use crate::polytope::Polytope;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeOp;
use crate::star_node::StarNodeType;
use crate::NNVFloat;
use ndarray::ArrayView1;
use ndarray::Dimension;
use ndarray::Ix2;

pub trait StarSet<D: Dimension> {
    fn get_node(&self, node_id: usize) -> &StarNode<D>;
    fn get_node_mut(&mut self, node_id: usize) -> &mut StarNode<D>;
    fn add_node(&mut self, node: StarNode<D>, parent_id: usize) -> usize;
    fn get_dnn(&self) -> &DNN;
    fn try_get_node_parent_id(&self, node_id: usize) -> Option<usize>;
    fn get_node_dnn_index(&self, node_id: usize) -> DNNIndex;
    fn try_get_node_type(&self, node_id: usize) -> &Option<StarNodeType>;
    fn set_node_type(&mut self, node_id: usize, children: StarNodeType);
    fn reset_with_star(&mut self, input_star: Star<D>);

    fn get_root_id(&self) -> usize {
        0
    }

    fn is_node_leaf(&mut self, node_id: usize) -> bool {
        *self.try_get_node_type(node_id) == Some(StarNodeType::Leaf)
    }

    /// # Panics
    fn get_node_ancestors(&self, node_id: usize) -> Vec<usize> {
        let mut ancestors = vec![];
        let mut current_node = Some(node_id);
        while let Some(idx) = current_node {
            ancestors.push(idx);
            current_node = self.try_get_node_parent_id(idx);
        }
        ancestors
    }
}

pub trait StarSet2: StarSet<Ix2> {
    fn get_node_type(&mut self, node_id: usize) -> &StarNodeType;

    fn get_node_input_bounds(&self, node_id: usize) -> Option<&Bounds1> {
        self.get_node(node_id).get_input_bounds()
    }

    fn get_node_reduced_input_polytope(&self, node_id: usize) -> Option<Polytope> {
        self.get_node(node_id).get_reduced_input_polytope()
    }

    fn is_node_member(&self, node_id: usize, point: &ArrayView1<NNVFloat>) -> bool {
        self.get_node(node_id).is_input_member(point)
    }

    fn get_node_child_ids(&mut self, node_id: usize) -> Vec<usize> {
        self.get_node_type(node_id).get_child_ids()
    }

    fn expand(&mut self, node_id: usize) -> &StarNodeType {
        let dnn_index = self.get_node_dnn_index(node_id);
        let dnn_iter = &mut DNNIterator::new(self.get_dnn(), dnn_index);

        // Get this node's operation from the dnn_iter
        let op = dnn_iter.next();
        // Do this node's operation to produce its children
        let children = match op {
            Some(StarNodeOp::Leaf) => StarNodeType::Leaf,
            Some(StarNodeOp::Affine(aff)) => {
                let child_idx = self.add_node(
                    StarNode::default(self.get_node(node_id).get_star().affine_map2(&aff), None)
                        .with_dnn_index(dnn_iter.get_idx()),
                    node_id,
                );
                StarNodeType::Affine { child_idx }
            }
            Some(StarNodeOp::StepRelu(dim)) => {
                let child_stars = self.get_node(node_id).get_star().step_relu2(dim);
                let dnn_idx = dnn_iter.get_idx();

                let mut ids = vec![];

                if let Some(lower_star) = child_stars.0 {
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .calculate_star_local_bounds()
                        .clone();
                    bounds.index_mut(dim)[0] = 0.;
                    bounds.index_mut(dim)[1] = 0.;
                    let lower_node =
                        StarNode::default(lower_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(lower_node, node_id);
                    ids.push(id);
                }

                if let Some(upper_star) = child_stars.1 {
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .calculate_star_local_bounds()
                        .clone();
                    let mut lb = bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = 0.;
                    }
                    let upper_node =
                        StarNode::default(upper_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(upper_node, node_id);
                    ids.push(id);
                }

                StarNodeType::StepRelu {
                    dim,
                    fst_child_idx: ids[0],
                    snd_child_idx: ids.get(1).copied(),
                }
            }
            Some(StarNodeOp::StepReluDropout((dropout_prob, dim))) => {
                let child_stars = self.get_node(node_id).get_star().step_relu2_dropout(dim);
                let dnn_idx = dnn_iter.get_idx();
                let mut ids = vec![];

                if let Some(dropout_star) = child_stars.0 {
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .calculate_star_local_bounds()
                        .clone();
                    bounds.index_mut(dim)[0] = 0.;
                    bounds.index_mut(dim)[1] = 0.;
                    let dropout_node =
                        StarNode::default(dropout_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(dropout_node, node_id);
                    ids.push(id);
                }

                if let Some(lower_star) = child_stars.1 {
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .calculate_star_local_bounds()
                        .clone();
                    bounds.index_mut(dim)[0] = 0.;
                    bounds.index_mut(dim)[1] = 0.;
                    let lower_node =
                        StarNode::default(lower_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(lower_node, node_id);
                    ids.push(id);
                }

                if let Some(upper_star) = child_stars.2 {
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .calculate_star_local_bounds()
                        .clone();
                    let mut lb = bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = 0.;
                    }
                    let upper_node =
                        StarNode::default(upper_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(upper_node, node_id);
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
        self.set_node_type(node_id, children);
        self.get_node_type(node_id)
    }
}

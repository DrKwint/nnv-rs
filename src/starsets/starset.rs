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
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::Dimension;
use ndarray::Ix2;

pub trait StarSet<D: 'static + Dimension> {
    type NI<'a>: Iterator<Item = &'a StarNode<D>>
    where
        Self: 'a,
        D: 'a;
    fn get_node(&self, node_id: usize) -> &StarNode<D>;
    fn get_node_mut(&mut self, node_id: usize) -> &mut StarNode<D>;
    fn get_node_iter(&self) -> Self::NI<'_>;
    fn add_node(&mut self, node: StarNode<D>, parent_id: usize) -> usize;
    fn get_dnn(&self) -> &DNN;
    fn get_input_bounds(&self) -> &Option<Bounds1>;
    fn try_get_node_parent_id(&self, node_id: usize) -> Option<usize>;
    fn get_node_dnn_index(&self, node_id: usize) -> DNNIndex;
    fn try_get_node_type(&self, node_id: usize) -> &Option<StarNodeType>;
    fn set_node_type(&mut self, node_id: usize, children: StarNodeType);
    fn reset_with_star(&mut self, input_star: Star<D>, input_bounds_opt: Option<Bounds1>);

    fn get_root_id(&self) -> usize {
        0
    }

    fn is_node_leaf(&self, node_id: usize) -> bool {
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

    fn get_node_reduced_input_polytope(&self, node_id: usize) -> Option<Polytope> {
        self.get_node(node_id)
            .get_reduced_input_polytope(self.get_input_bounds())
    }

    fn is_node_member(&self, node_id: usize, point: &ArrayView1<NNVFloat>) -> bool {
        self.get_node(node_id).is_input_member(point)
    }

    fn get_node_child_ids(&mut self, node_id: usize) -> Vec<usize> {
        self.get_node_type(node_id).get_child_ids()
    }

    fn can_node_maximize_output_idx(&self, node_id: usize, class_idx: usize) -> bool {
        self.get_node(node_id)
            .get_star()
            .can_maximize_output_idx(class_idx)
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
                let child_stars = self
                    .get_node(node_id)
                    .get_star()
                    .step_relu2(dim, self.get_input_bounds());
                let dnn_idx = dnn_iter.get_idx();

                let mut ids = vec![];

                let is_single_child = child_stars.0.is_some() ^ child_stars.1.is_some();

                if let Some(mut lower_star) = child_stars.0 {
                    let outer_bounds: Bounds1 = self.get_input_bounds().as_ref().cloned().unwrap();
                    let mut input_bounds = self
                        .get_node_mut(node_id)
                        .get_axis_aligned_input_bounds(&outer_bounds)
                        .clone();
                    input_bounds.index_mut(dim)[0] = 0.;
                    input_bounds.index_mut(dim)[1] = 0.;
                    if is_single_child {
                        // Remove redundant constraint added by step_relu2 above
                        let num_constraints = lower_star.num_constraints();
                        lower_star = lower_star.remove_constraint(num_constraints - 1);
                    }
                    let mut lower_node =
                        StarNode::default(lower_star, Some(input_bounds)).with_dnn_index(dnn_idx);

                    if is_single_child {
                        if let Some(cdf) = self.get_node(node_id).try_get_cdf() {
                            lower_node.set_cdf(cdf);
                        }
                        if let Some(dist) = self.get_node(node_id).try_get_gaussian_distribution() {
                            lower_node.set_gaussian_distribution(dist.clone());
                        }
                    }
                    let id = self.add_node(lower_node, node_id);
                    ids.push(id);
                }

                if let Some(mut upper_star) = child_stars.1 {
                    let outer_bounds: Bounds1 = self.get_input_bounds().as_ref().cloned().unwrap();
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .get_axis_aligned_input_bounds(&outer_bounds)
                        .clone();
                    let mut lb = bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = 0.;
                    }
                    if is_single_child {
                        // Remove redundant constraint added by step_relu2 above
                        let num_constraints = upper_star.num_constraints();
                        upper_star = upper_star.remove_constraint(num_constraints - 1);
                    }
                    let mut upper_node =
                        StarNode::default(upper_star, Some(bounds)).with_dnn_index(dnn_idx);

                    if is_single_child {
                        if let Some(cdf) = self.get_node(node_id).try_get_cdf() {
                            upper_node.set_cdf(cdf);
                        }
                        if let Some(bounds) = self.get_node(node_id).try_get_output_bounds() {
                            upper_node.set_output_bounds(bounds);
                        }
                        if let Some(dist) = self.get_node(node_id).try_get_gaussian_distribution() {
                            upper_node.set_gaussian_distribution(dist.clone());
                        }
                    }
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
                let child_stars = self
                    .get_node(node_id)
                    .get_star()
                    .step_relu2_dropout(dim, self.get_input_bounds());
                let dnn_idx = dnn_iter.get_idx();
                let mut ids = vec![];

                if let Some(dropout_star) = child_stars.0 {
                    let outer_bounds: Bounds1 = self.get_input_bounds().as_ref().cloned().unwrap();
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .get_axis_aligned_input_bounds(&outer_bounds)
                        .clone();
                    bounds.index_mut(dim)[0] = 0.;
                    bounds.index_mut(dim)[1] = 0.;
                    let dropout_node =
                        StarNode::default(dropout_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(dropout_node, node_id);
                    ids.push(id);
                }

                if let Some(lower_star) = child_stars.1 {
                    let outer_bounds: Bounds1 = self.get_input_bounds().as_ref().cloned().unwrap();
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .get_axis_aligned_input_bounds(&outer_bounds)
                        .clone();
                    bounds.index_mut(dim)[0] = 0.;
                    bounds.index_mut(dim)[1] = 0.;
                    let lower_node =
                        StarNode::default(lower_star, Some(bounds)).with_dnn_index(dnn_idx);
                    let id = self.add_node(lower_node, node_id);
                    ids.push(id);
                }

                if let Some(upper_star) = child_stars.2 {
                    let outer_bounds: Bounds1 = self.get_input_bounds().as_ref().cloned().unwrap();
                    let mut bounds = self
                        .get_node_mut(node_id)
                        .get_axis_aligned_input_bounds(&outer_bounds)
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

    fn run_datum_to_leaf(&mut self, datum: &Array1<NNVFloat>) -> usize {
        // Run through the input datum
        let mut current_node_id = self.get_root_id();
        let mut current_node_type = self.get_node_type(current_node_id).clone();
        let activation_pattern = self.get_dnn().calculate_activation_pattern1(datum);
        // For each ReLU layer activation pattern
        for layer_activations in &activation_pattern {
            // Go through the Affine
            if let StarNodeType::Affine { child_idx } = current_node_type {
                current_node_id = child_idx;
            }
            current_node_type = self.get_node_type(current_node_id).clone();
            // For each activation
            for activation in layer_activations {
                // Select a child node based on the activation
                if let StarNodeType::StepRelu {
                    dim,
                    fst_child_idx,
                    snd_child_idx,
                } = current_node_type
                {
                    if *activation {
                        current_node_id = fst_child_idx;
                    } else {
                        current_node_id = snd_child_idx.expect("Error selecting a second child!");
                    }
                } else {
                    panic!("Expected a ReLU layer!");
                }
                current_node_type = self.get_node_type(current_node_id).clone();
            }
        }
        if let StarNodeType::Affine { child_idx } = current_node_type {
            current_node_id = child_idx;
            // current_node_type = self.get_node_type(current_node_id).clone();
        }
        current_node_id
    }
}

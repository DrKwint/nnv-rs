use std::iter;

use crate::bounds::Bounds1;
use crate::dnn::dnn::DNN;
use crate::dnn::dnn_iter::DNNIndex;
use crate::dnn::dnn_iter::DNNIterator;
use crate::polytope::Polytope;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeType;
use crate::NNVFloat;
use ndarray::concatenate;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::Ix2;

pub trait StarSet<D: 'static + Dimension> {
    type NI<'a>: Iterator<Item = &'a StarNode<D>>
    where
        Self: 'a,
        D: 'a;

    type NTI<'a>: Iterator<Item = &'a Option<StarNodeType>>
    where
        Self: 'a,
        D: 'a;

    fn get_node(&self, node_id: usize) -> &StarNode<D>;
    fn get_node_mut(&mut self, node_id: usize) -> &mut StarNode<D>;
    fn get_node_iter(&self) -> Self::NI<'_>;
    fn get_node_type_iter(&self) -> Self::NTI<'_>;
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
        match *self.try_get_node_type(node_id) {
            Some(StarNodeType::Leaf { .. }) => true,
            _ => false,
        }
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
    fn get_node_type_mut(&mut self, node_id: usize) -> &mut StarNodeType;

    fn get_node_reduced_input_polytope(&self, node_id: usize) -> Option<Polytope> {
        self.get_node(node_id)
            .get_reduced_input_polytope(self.get_input_bounds())
    }

    fn is_node_member(&self, node_id: usize, point: &ArrayView1<NNVFloat>) -> bool {
        self.get_node(node_id).is_input_member(point)
    }

    fn filter_node_member(
        &self,
        node_id: usize,
        points: Vec<Array1<NNVFloat>>,
    ) -> Vec<Array1<NNVFloat>> {
        let unsafe_len = points[0].len();
        let sample_iter = points.into_iter();
        let fixed_input_part: Option<Array1<NNVFloat>> = {
            self.get_input_bounds().as_ref().map(|bounds| {
                let fixed_bounds: Bounds1 = bounds.split_at(bounds.ndim() - unsafe_len).0;
                let fixed_array: Array1<NNVFloat> = fixed_bounds.lower().to_owned();
                fixed_array
            })
        };
        let node = self.get_node(node_id);
        let bounds = self.get_input_bounds().as_ref();
        if let Some(fix_part) = fixed_input_part {
            sample_iter
                .zip(iter::repeat(fix_part))
                .map(|(unfix, fix)| {
                    (
                        unfix.clone(),
                        concatenate(Axis(0), &[fix.view(), unfix.view()]).unwrap(),
                    )
                })
                .filter(|(_sample, x)| bounds.map(|b| b.is_member(&x.view())).unwrap_or(true))
                .filter(|(_sample, x)| node.is_input_member(&x.view()))
                .map(|(sample, _)| sample)
                .collect()
        } else {
            sample_iter
                .filter(|sample| bounds.map(|b| b.is_member(&sample.view())).unwrap_or(true))
                .filter(|sample| node.is_input_member(&sample.view()))
                .collect()
        }
    }

    fn get_node_child_ids(&mut self, node_id: usize) -> Vec<usize> {
        self.get_node_type(node_id).get_child_ids()
    }

    fn can_node_maximize_output_idx(&self, node_id: usize, class_idx: usize) -> bool {
        self.get_node(node_id)
            .get_star()
            .can_maximize_output_idx(class_idx)
    }

    fn expand(&mut self, node_id: usize) -> &mut StarNodeType {
        let dnn_index = self.get_node_dnn_index(node_id);
        let dnn_iter = &mut DNNIterator::new(self.get_dnn(), dnn_index);

        // Get this node's operation from the dnn_iter
        let op_idx = dnn_iter.next();

        // Do this node's operation to produce its children
        let children = match op_idx {
            None => StarNodeType::Leaf {
                safe_idx: None,
                unsafe_idx: None,
            },
            Some(op_idx) => {
                let (child_stars, star_input_bounds, same_output_bounds) = match op_idx {
                    DNNIndex {
                        layer: Some(idx),
                        remaining_steps: Some(dim),
                    } => {
                        let outer_bounds = &self.get_input_bounds().as_ref().cloned().unwrap();
                        let parent_bounds = self
                            .get_node_mut(node_id)
                            .get_axis_aligned_input_bounds(outer_bounds)
                            .clone();
                        let star = self.get_node(node_id).get_star();
                        let bounds = self.get_input_bounds().clone();
                        self.get_dnn().get_layer(idx).unwrap().forward_star(
                            star,
                            Some(dim),
                            bounds,
                            Some(parent_bounds),
                        )
                    }
                    DNNIndex {
                        layer: Some(idx), ..
                    } => {
                        let layer = self.get_dnn().get_layer(idx).unwrap();
                        layer.forward_star(self.get_node(node_id).get_star(), None, None, None)
                    }
                    DNNIndex { layer: None, .. } => panic!(),
                };
                debug_assert!(!child_stars.is_empty());
                debug_assert!(child_stars.len() == star_input_bounds.len());
                let mut child_ids = vec![];
                for (star, input_bounds) in
                    child_stars.into_iter().zip(star_input_bounds.into_iter())
                {
                    let node = StarNode::default(star, input_bounds, op_idx);
                    let id = self.add_node(node, node_id);
                    child_ids.push(id);
                }

                if child_ids.len() == 1 {
                    if let Some(cdf) = self.get_node(node_id).try_get_cdf() {
                        self.get_node_mut(child_ids[0]).set_cdf(cdf);
                    }
                    if let Some(dist) = self.get_node(node_id).try_get_gaussian_distribution() {
                        let dist = dist.clone();
                        self.get_node_mut(child_ids[0])
                            .set_gaussian_distribution(dist);
                    }

                    if same_output_bounds {
                        if let Some(bounds) = self.get_node(node_id).try_get_output_bounds() {
                            self.get_node_mut(child_ids[0]).set_output_bounds(bounds);
                        }
                    }
                }
                self.get_dnn()
                    .get_layer(op_idx.layer.unwrap())
                    .unwrap()
                    .construct_starnodetype(&child_ids, op_idx.get_remaining_steps())
            }
        };

        self.set_node_type(node_id, children);
        self.get_node_type_mut(node_id)
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
                    dim: _,
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

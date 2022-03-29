use std::collections::HashSet;
use std::iter;

use crate::bounds::Bounds1;
use crate::dnn::dnn::DNN;
use crate::graph::Operation;
use crate::graph::OperationId;
use crate::graph::OperationNode;
use crate::polytope::Polytope;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeRelationship;
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
    fn get_parent_nodes(&self, node_id: usize) -> Vec<&StarNode<D>>;
    fn get_child_nodes(&self, node_id: usize) -> Option<Vec<&StarNode<D>>>;
    fn get_parent_ids(&self, node_id: usize) -> Vec<usize>;
    fn get_child_ids(&self, node_id: usize) -> Option<Vec<usize>>;
    // fn get_node_iter(&self) -> Self::NI<'_>;
    // fn get_node_type_iter(&self) -> Self::NTI<'_>;
    fn get_creating_relationship(&self, node_id: usize) -> Option<&StarNodeRelationship>;
    fn add_node(&mut self, node: StarNode<D>) -> usize;
    fn add_node_relationship(&mut self, rel: StarNodeRelationship) -> usize;
    fn get_node_relationship(&self, rel_id: usize) -> &StarNodeRelationship;
    fn get_node_relationship_mut(&self, rel_id: usize) -> &mut StarNodeRelationship;
    fn get_node_producing_relationship_id(&self, node_id: usize) -> usize;
    fn get_dnn(&self) -> &DNN;
    fn get_operation(&self, op_id: &OperationId) -> &dyn Operation;
    fn get_operation_node(&self, op_id: &OperationId) -> &OperationNode;

    /// If you have no bounds for a specific input, use the trivial bounds.
    fn get_input_bounds(&self) -> &Option<Vec<Bounds1>>;
    fn try_get_node_parent_ids(&self, node_id: usize) -> Option<&Vec<usize>>;
    fn reset_with_star(&mut self, input_star: Star<D>, input_bounds_opt: Option<Bounds1>);

    fn get_root_id(&self) -> usize {
        0
    }

    /// A leaf node returns `Some(vec)` to `get_children` where `vec.is_empty() == true`,
    /// i.e. all non-leaf nodes have at least one child. Returns None if the node needs to
    /// first be expanded before testing.
    fn try_is_node_leaf(&self, node_id: usize) -> Option<bool> {
        self.get_child_ids(node_id)
            .map(|child_ids| child_ids.is_empty())
    }

    /// Returns a set of node ids that are all ancestors of `node_id`
    /// # Panics
    fn get_node_ancestors(&self, node_id: usize) -> HashSet<usize> {
        let mut ancestors = HashSet::new();
        let mut parent_stack = vec![node_id];
        while let Some(current_id) = parent_stack.pop() {
            if let Some(parent_ids) = self.try_get_node_parent_ids(current_id) {
                parent_ids.into_iter().for_each(|&parent_id| {
                    if !ancestors.contains(&parent_id) {
                        parent_stack.push(parent_id);
                        ancestors.insert(parent_id);
                    }
                })
            }
        }
        ancestors
    }
}

pub trait StarSet2: StarSet<Ix2> {
    // fn get_node_type(&mut self, node_id: usize) -> &StarNodeType;
    // fn get_node_type_mut(&mut self, node_id: usize) -> &mut StarNodeType;

    fn get_node_reduced_input_polytope(&self, node_id: usize) -> Option<Polytope> {
        self.get_node(node_id)
            .get_reduced_input_polytope(self.get_input_bounds())
    }

    /// Checks if `point` is a member of the input space of the star contained by the node
    fn is_node_member(&self, node_id: usize, point: &ArrayView1<NNVFloat>) -> bool {
        self.get_node(node_id).is_input_member(point)
    }

    /// Filters points to only those that are a member of the input set of the node corresponding to `node_id`.
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

    fn can_node_maximize_output_idx(&self, node_id: usize, class_idx: usize) -> bool {
        self.get_node(node_id)
            .get_star()
            .can_maximize_output_idx(class_idx)
    }

    /// Expands an operation from it's inputs to produce the children
    ///
    /// # Description
    /// Each child node is stored as a separate StarNode in the StarSet. If an operation has multiple steps
    fn expand(
        &mut self,
        operation_id: OperationId,
        input_node_ids: Vec<usize>,
    ) -> &mut StarNodeRelationship {
        let op_node = self.get_operation_node(&operation_id);
        assert_eq!(op_node.get_input_ids().len(), input_node_ids.len());
        // let dnn_index = self.get_node_dnn_index(node_id);
        // let dnn_iter = &mut DNNIterator::new(self.get_dnn(), dnn_index);

        // Get this node's operation from the dnn_iter
        // let op_idx = dnn_iter.next();

        let mut operation_step = None;

        if let Some(rel) = self.get_creating_relationship(*input_node_ids.first().unwrap()) {
            if let Some(step) = rel.step {
                assert_eq!(operation_id, rel.operation_id);
                let op = self.get_operation(&rel.operation_id);
                if let Some(num_steps) = op.num_steps() {
                    // Continue a stepped operation
                    operation_step = if step + 2 < num_steps {
                        Some(step + 1)
                    } else {
                        None
                    };
                }
            }
        }

        let outer_bounds = &self.get_input_bounds().as_ref().cloned().unwrap();
        let parent_bounds = input_node_ids
            .iter()
            .map(|&node_id| {
                self.get_node_mut(node_id)
                    .get_axis_aligned_input_bounds(outer_bounds)
                    .clone()
            })
            .collect::<Vec<_>>();
        let stars = input_node_ids
            .iter()
            .map(|&node_id| self.get_node(node_id).get_star())
            .collect::<Vec<_>>();
        let input_bounds = self.get_input_bounds().clone();

        let (child_stars, child_input_bounds, same_output_bounds) = op_node
            .get_operation()
            .forward_star(stars, operation_step, input_bounds, Some(parent_bounds));

        debug_assert!(!child_stars.is_empty());
        debug_assert!(child_stars.len() == child_input_bounds.len());
        let mut child_ids = vec![];
        for (star, input_bounds) in child_stars.into_iter().zip(child_input_bounds.into_iter()) {
            let node = StarNode::default(star, input_bounds);
            let id = self.add_node(node);
            child_ids.push(id);
        }

        if child_ids.len() == 1 && input_node_ids.len() == 1 {
            let &parent_id = input_node_ids.first().unwrap();
            if let Some(cdf) = self.get_node(parent_id).try_get_cdf() {
                self.get_node_mut(child_ids[0]).set_cdf(cdf);
            }
            if let Some(dist) = self.get_node(parent_id).try_get_gaussian_distribution() {
                let dist = dist.clone();
                self.get_node_mut(child_ids[0])
                    .set_gaussian_distribution(dist);
            }

            if same_output_bounds {
                if let Some(bounds) = self.get_node(parent_id).try_get_output_bounds() {
                    self.get_node_mut(child_ids[0]).set_output_bounds(bounds);
                }
            }
        }

        let rel = StarNodeRelationship {
            operation_id,
            step: operation_step,
            input_node_ids,
            output_node_ids: Some(child_ids),
        };
        let rel_id = self.add_node_relationship(rel);
        self.get_node_relationship_mut(rel_id)
    }

    fn run_datum_to_leaf(&mut self, datum: &Array1<NNVFloat>) -> usize {
        todo!();
        //     // Run through the input datum
        //     let mut current_node_id = self.get_root_id();
        //     let mut current_node_type = self.get_node_type(current_node_id).clone();
        //     let activation_pattern = self.get_dnn().calculate_activation_pattern1(datum);
        //     // For each ReLU layer activation pattern
        //     for layer_activations in &activation_pattern {
        //         // Go through the Affine
        //         if let StarNodeType::Affine { child_idx } = current_node_type {
        //             current_node_id = child_idx;
        //         }
        //         current_node_type = self.get_node_type(current_node_id).clone();
        //         // For each activation
        //         for activation in layer_activations {
        //             // Select a child node based on the activation
        //             if let StarNodeType::StepRelu {
        //                 dim: _,
        //                 fst_child_idx,
        //                 snd_child_idx,
        //             } = current_node_type
        //             {
        //                 if *activation {
        //                     current_node_id = fst_child_idx;
        //                 } else {
        //                     current_node_id = snd_child_idx.expect("Error selecting a second child!");
        //                 }
        //             } else {
        //                 panic!("Expected a ReLU layer!");
        //             }
        //             current_node_type = self.get_node_type(current_node_id).clone();
        //         }
        //     }
        //     if let StarNodeType::Affine { child_idx } = current_node_type {
        //         current_node_id = child_idx;
        //         // current_node_type = self.get_node_type(current_node_id).clone();
        //     }
        //     current_node_id
    }
}

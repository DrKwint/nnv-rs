use crate::bounds::Bounds;
use crate::bounds::Bounds1;
use crate::dnn::DNN;
use crate::graph::Graph;
use crate::graph::Operation;
use crate::graph::OperationId;
use crate::graph::RepresentationId;
use crate::star::Star;
use crate::util::get_next_step;
use ndarray::Dimension;
use ndarray::Ix2;
use serde::{Deserialize, Serialize};
use std::cell::Ref;

pub type StarId = usize;
pub type StarRelationshipId = usize;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StarRelationship {
    pub operation_id: OperationId,
    pub step: Option<usize>,
    pub input_star_ids: Vec<usize>,
    pub output_star_ids: Vec<Vec<usize>>,
}

/// We assume there's a root star that is the ordered concatenation of the DNN's input variables.
/// This facilitates the representation of each star.
pub trait StarSet<D: 'static + Dimension> {
    /// Get the Graph from the DNN
    fn get_graph(&self) -> &Graph;
    /// Get DNN
    fn get_dnn(&self) -> &DNN;
    /// Get the input bounds of the star set
    fn get_input_bounds(&self) -> &Bounds<D>;
    /// Get the id of the root star in the starset.
    fn get_root_id(&self) -> StarId;
    /// Get the DNN/Graph `RepresentationId` that a star corresponds to
    fn get_star_representation_id(&self, star_id: StarId) -> RepresentationId;
    /// Get a reference to a star
    fn get_star(&self, star_id: StarId) -> Ref<Star<D>>;
    /// Gets a relationship
    fn get_relationship(&self, relationship_id: StarRelationshipId) -> Ref<StarRelationship>;
    /// Add a star
    /// Requires interior mutability
    fn add_star(
        &self,
        star: Star<D>,
        representation_id: RepresentationId,
        local_output_bounds: Option<Bounds<D>>,
    ) -> StarId;
    /// Adds a relationship
    /// Requires interior mutability
    fn add_relationship(&self, star_rel: StarRelationship) -> StarRelationshipId;
}

pub trait StarSet2: StarSet<Ix2> {
    //! # TODO: discuss the API of this trait
    //! # TODO: discuss assumptions of this trait

    /// Get the dimension of the DNN input
    fn get_input_dim(&self) -> usize;
    /// TODO: Implement with a cache because it is expensive
    fn get_local_output_bounds(&self, star_id: StarId) -> Ref<Option<Bounds1>>;

    /// Expand an operation from its inputs to produce the children and adds them to the `StarSet`.
    ///
    /// # Description
    ///
    /// Each non-empty child star is stored as a separate `StarNode` in the `StarSet`.
    ///
    /// # Invariants
    ///
    /// The stars pointed to by `input_stars_ids` must be those that correspond to the `RepresentationId`s of the inputs to the operation and they must be in the same order.
    ///
    /// # Arguments
    /// * `operation_id` - The operation of the DNN on which to expand the star set.
    /// * `input_star_ids` - The ordered ids of the `star`s that are used as inputs to the operation.
    fn expand(&self, operation_id: OperationId, input_star_ids: Vec<StarId>) -> StarRelationshipId {
        // Pre-condition asserts
        assert!(!input_star_ids.is_empty());
        let operation_node = self.get_graph().get_operation_node(&operation_id).unwrap();

        // Check if the `input_star_ids` map to the correct `representation_id`s
        // TODO: Reorder the inputs according to the operation node.
        let repr_ids = input_star_ids
            .iter()
            .map(|star_id| self.get_star_representation_id(*star_id))
            .collect::<Vec<_>>();
        // Comparison checks ordering of `repr_ids`, and by extension, `input_star_ids`
        assert_eq!(repr_ids.len(), operation_node.get_input_ids().len());

        // Make sure the step of each input star's `RepresentationId` is the same.
        // Non-empty assert makes the unwrap safe.
        let prev_step = repr_ids.first().unwrap().operation_step;
        assert!(!repr_ids
            .into_iter()
            .any(|repr_id| repr_id.operation_step != prev_step));

        // Calculate next step
        let (repr_step, step) =
            get_next_step(operation_node.get_operation().num_steps(), prev_step);

        // 1. Calculate output stars
        let outputs = {
            let stars: Vec<_> = input_star_ids
                .iter()
                .map(|&node_id| self.get_star(node_id))
                .collect();

            let local_storage: Vec<_> = input_star_ids
                .iter()
                .map(|&star_id| Ref::map(self.get_local_output_bounds(star_id), |x| x))
                .collect();
            let parent_local_output_bounds: Option<Vec<_>> =
                local_storage.iter().map(|x| x.as_ref()).collect();

            operation_node.get_operation().forward_star(
                stars,
                step,
                self.get_input_bounds(),
                parent_local_output_bounds,
            )
        };

        // 2. Add child stars and StarRelationship
        let child_star_ids: Vec<Vec<usize>> = outputs
            .into_iter()
            .zip(operation_node.get_output_ids().iter())
            .map(
                |((child_stars, child_input_bounds), &output_repr_id)| -> Vec<usize> {
                    child_stars
                        .into_iter()
                        .zip(child_input_bounds.into_iter())
                        .map(|(star, child_input_bounds)| {
                            let out_repr_id = output_repr_id.clone().with_step(repr_step);
                            self.add_star(star, out_repr_id, child_input_bounds)
                        })
                        .collect()
                },
            )
            .collect();

        // let child_star_ids = child_stars
        //     .into_iter()
        //     .zip(operation_node.get_output_ids().clone().into_iter())
        //     .zip(child_input_bounds.into_iter())
        //     .map(|((star, repr_id), child_input_bounds)| {
        //         self.add_star(star, repr_id, child_input_bounds)
        //     })
        //     .collect();
        let star_rel = StarRelationship {
            operation_id,
            step: repr_step,
            input_star_ids,
            output_star_ids: child_star_ids,
        };
        self.add_relationship(star_rel)
    }
}

#[cfg(test)]
mod tests {
    use super::super::new_graph_starset::GraphStarset;
    use super::*;
    use crate::dnn::DNN;
    use crate::graph::Engine;
    use crate::star::Star2;
    use crate::test_util::*;
    use proptest::prelude::*;

    #[must_use = "strategies do nothing unless used"]
    fn generic_test_inputs(
        max_input_dim: usize,
        max_output_dim: usize,
        max_n_hidden_layers: usize,
        max_layer_width: usize,
        max_num_constraints: usize,
    ) -> impl Strategy<Value = (DNN, Star2, Bounds1)> {
        let strat = (
            1..(max_input_dim + 1),
            1..(max_output_dim + 1),
            1..(max_n_hidden_layers + 1),
            0..(max_num_constraints + 1),
        );
        Strategy::prop_flat_map(
            strat,
            move |(input_dim, output_dim, n_hidden_layers, num_constraints)| {
                (
                    fc_dnn(input_dim, output_dim, n_hidden_layers, max_layer_width),
                    non_empty_star(input_dim, num_constraints),
                    bounds1(input_dim),
                )
            },
        )
    }

    proptest! {
        #[test]
        fn test_expand((dnn, input_star, input_bounds) in generic_test_inputs(2,2,2,2,0)) {
            let starset = GraphStarset::new(dnn, input_bounds, input_star);

            // First operation is a dense
            let rel_id = starset.expand(0, vec![starset.get_root_id()]);
            let rel = starset.get_relationship(rel_id);
            prop_assert_eq!(0, rel.operation_id);
            prop_assert_eq!(None, rel.step);
            prop_assert_eq!(rel.input_star_ids.len(), 1);
            prop_assert_eq!(rel.output_star_ids.len(), 1);
            prop_assert!(rel.output_star_ids[0][0] > rel.input_star_ids[0]);
        }

        #[test]
        fn test_expand_whole_tree((dnn, input_star, input_bounds) in generic_test_inputs(2,2,2,2,0)) {
            let starset = GraphStarset::new(dnn, input_bounds, input_star);
            let engine = Engine::new(starset.get_graph());

            let num_steps = starset.get_graph().get_operations().into_iter().fold(0, |acc, x| acc +
                x.get_operation().num_steps().unwrap_or(0)
            );

            // Expand all nodes in the starset tree
            // 1. Create a frontier
            // let frontier = vec![(starset.get_dnn().get_input_representation_ids()[0].clone(), starset.get_root_id())];


            // 2. Visit each operation in order
            let inputs = vec![(starset.get_dnn().get_input_representation_ids()[0].clone(), vec![starset.get_root_id()])];
            let res = engine.run_nodal(starset.get_dnn().get_output_representation_ids(), &inputs, |op_id, op_node, inputs, step| -> (Option<usize>, Vec<Vec<usize>>) {
                assert_eq!(1, inputs.len());
                let (repr_step, _next_step) = get_next_step(op_node.get_operation().num_steps(), step);
                let input_stars = inputs[0];
                let star_ids = input_stars.into_iter().map(|input_star_id| {
                    let rel_id = starset.expand(op_id, vec![*input_star_id]);
                    let rel = starset.get_relationship(rel_id);
                    assert_eq!(repr_step, rel.step);
                    rel.output_star_ids.clone().into_iter().flatten()
                }).flatten().collect();

                (repr_step, vec![star_ids])
            });

            prop_assert!(res.is_ok(), "{:?}", res);
            let res = res.unwrap();
            prop_assert_eq!(res.len(), 1);
            let (_repr_id, out_stars) = &res[0];
            prop_assert!(out_stars.len() <= usize::pow(2, num_steps as u32));
        }
    }
}

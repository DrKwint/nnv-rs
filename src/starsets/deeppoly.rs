use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::dnn::DNN;
use crate::graph::{Engine, Operation, OperationId, OperationNode, PhysicalOp, RepresentationId};
use crate::util::get_next_step;
use itertools::Itertools;
use ndarray::{Array1, Array2, Axis, Slice};

/// Calculates the output bounds of the output representations of the suffix `dnn` starting from `input_nodes`
///
/// # Description
///
/// Let `l = \sum_k_i`, where `k_i = |bounds_i|`.
/// Steps:
///     1. Create lower and upper input affines for each of the inputs of size `l x k_i`.
///        The weights of the affines are identity blocks at `\sum_{j\in i-1} k_j` surrounded by zero blocks with zero bias.
///     2. For each operation, calculate the new upper and lower affines of the operation, each having shape `l x k_j`, where `k_j = |bounds_j|`.
///     3. Concretize the final output bounds.
///
/// # Arguments
///
/// * `dnn` - A network to operate on
/// * `input_nodes` - Input node representations along with bounds for each input node.
///
pub fn deep_poly(
    dnn: &DNN,
    input_nodes: &Vec<(RepresentationId, &Bounds1)>,
    output_ids: &Vec<RepresentationId>,
) -> Vec<Bounds1> {
    _deep_poly(dnn, input_nodes, output_ids).0
}

/// This function is used for testing deep_poly and should not be used in the public API.
/// In addition to the tighter bounds calculated by concretizing the abstract bounds with
/// the input bounds, this function also returns the concrete bounds, which is used to
/// ensure that the concrete bounds are in fact looser.
pub fn _deep_poly(
    dnn: &DNN,
    input_nodes: &Vec<(RepresentationId, &Bounds1)>,
    output_ids: &Vec<RepresentationId>,
) -> (Vec<Bounds1>, Vec<Bounds1>) {
    assert!(!input_nodes.is_empty());

    let input_representations: Vec<(RepresentationId, (Bounds1, Affine2, Affine2))> = {
        // Calculate total input size
        let input_size: usize = input_nodes.iter().map(|(_, bounds)| bounds.ndim()).sum();

        let all_lower_matrix = Array2::eye(input_size); // input_size x input_size
        let all_upper_matrix = Array2::eye(input_size); // input_size x input_size

        input_nodes
            .into_iter()
            .scan(0, |start_idx, &(id, bounds)| {
                let dim = bounds.ndim();
                let end_idx: usize = *start_idx + dim;
                let output = (
                    id,
                    (
                        bounds.clone(),
                        Affine2::new(
                            all_lower_matrix
                                .slice_axis(Axis(1), Slice::from(*start_idx..end_idx))
                                .to_owned(),
                            Array1::zeros(input_size),
                        ),
                        Affine2::new(
                            all_upper_matrix
                                .slice_axis(Axis(1), Slice::from(*start_idx..end_idx))
                                .to_owned(),
                            Array1::zeros(input_size),
                        ), // Affine does (input_size x k_i) left(?) matrix mul + input_size vector add
                    ),
                );
                *start_idx = end_idx;
                Some(output)
            })
            .collect()
    };

    let engine = Engine::new(dnn.get_graph());
    let outputs = engine.run_nodal(
        output_ids,
        &input_representations,
        |op_id: OperationId,
         op_node: &OperationNode,
         inputs: &Vec<&(Bounds1, Affine2, Affine2)>,
         op_step: Option<usize>|
         -> (Option<usize>, Vec<(Bounds1, Affine2, Affine2)>) {
            let mut op_step = op_step.clone();
            // op_step is None if nothing has run yet, output None as the step when the entire Op is done
            // This visitor, if a step is taken from None, should increment None -> 0 and then op.num_steps -> None
            let (bounds_concrete, laff, uaff): (Vec<_>, Vec<_>, Vec<_>) = inputs
                .into_iter()
                .map(|&tup| (&tup.0, &tup.1, &tup.2))
                .multiunzip();
            let output_id = output_ids
                .iter()
                .filter(|&out_id| out_id.operation_step.is_some())
                .find(|&repr_id| {
                    op_node
                        .get_output_ids()
                        .iter()
                        .find(|&op_repr_id| {
                            op_repr_id.representation_node_id == repr_id.representation_node_id
                        })
                        .is_some()
                });
            if op_step.is_some() || output_id.is_some() {
                let (repr_step, next_step) =
                    get_next_step(op_node.get_operation().num_steps(), op_step);
                let next_step = next_step.unwrap();
                (
                    repr_step,
                    op_node.get_operation().apply_bounds_step(
                        next_step,
                        &bounds_concrete,
                        &laff,
                        &uaff,
                    ),
                )
            } else {
                (
                    None,
                    op_node
                        .get_operation()
                        .apply_bounds(&bounds_concrete, &laff, &uaff),
                )
            }
        },
    );
    let outputs = outputs.unwrap();

    // Collect all input bounds into one bound
    let input_bounds = input_nodes
        .iter()
        .fold(Bounds1::default(0), |acc, &(_, b)| acc.append(&b));

    // Concretize the bounds in terms of the input bounds
    outputs
        .into_iter()
        .map(|(_, (conc_bounds, lower_aff, upper_aff))| {
            let lower_bounds = lower_aff.signed_apply(&input_bounds);
            let upper_bounds = upper_aff.signed_apply(&input_bounds);
            let bounds = Bounds1::new(lower_bounds.lower(), upper_bounds.upper());
            debug_assert!(bounds.bounds_iter().into_iter().all(|x| x[[0]] <= x[[1]]));
            (bounds, conc_bounds)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::dense::Dense;
    use crate::dnn::dnn::DNN;
    use crate::dnn::relu::ReLU;
    use crate::graph::PhysicalOp;
    use crate::star::Star2;
    use crate::starsets::graph_starset::GraphStarset;
    use crate::starsets::starset::{StarSet, StarSet2};
    use crate::tensorshape::TensorShape;
    use crate::{test_util::*, NNVFloat};
    use proptest::prelude::*;
    use proptest::sample::select;
    use std::collections::HashMap;

    #[test]
    fn test_deeppoly_concrete() {
        let aff1: Affine2 = Affine2::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]).insert_axis(Axis(0)),
            Array1::from_vec(vec![7.85]),
        );
        let dense1 = Dense::new(aff1);
        let relu1 = ReLU::new(1);
        let aff2 = Affine2::new(
            Array1::from_vec(vec![9.49, 0.0]).insert_axis(Axis(1)),
            Array1::from_vec(vec![0., 0.]),
        );
        let dense2 = Dense::new(aff2);
        let relu2 = ReLU::new(2);
        let _dnn = DNN::from_sequential(&vec![
            PhysicalOp::from(dense1),
            PhysicalOp::from(relu1),
            PhysicalOp::from(dense2),
            PhysicalOp::from(relu2),
        ]);
        let _bounds: Bounds1 = Bounds1::new(
            Array1::from_vec(vec![0.0, 0.0, 0.]).view(),
            Array1::zeros(3).view(),
        );
    }

    fn get_all_stepped_repr_ids(dnn: &DNN) -> Vec<RepresentationId> {
        let non_inputs = dnn
            .get_graph()
            .get_operations()
            .iter()
            .map(|op_node| {
                let mut stepped_reprs = vec![];
                if let Some(num_steps) = op_node.get_operation().num_steps() {
                    for step in 0..(num_steps - 1) {
                        // Avoids underflow
                        if step == num_steps - 1 {
                            continue;
                        }

                        stepped_reprs
                            .push(op_node.get_output_ids()[0].clone().with_step(Some(step)));
                    }
                }
                stepped_reprs.push(op_node.get_output_ids()[0].clone());
                stepped_reprs.into_iter()
            })
            .flatten();
        dnn.get_input_representation_ids()
            .clone()
            .into_iter()
            .chain(non_inputs)
            .collect()
    }

    // Need to output a starting repr_id, ending repr_id with steps per repr_id
    fn fcdnn_with_star_end_repr(
        input_size: usize,
        output_size: usize,
        n_hidden_layers: usize,
        max_layer_width: usize,
    ) -> impl Strategy<Value = (DNN, Vec<RepresentationId>)> {
        let strat = (fc_dnn(
            input_size,
            output_size,
            n_hidden_layers,
            max_layer_width,
        ),);
        Strategy::prop_flat_map(strat, move |(dnn,)| {
            let operation_nodes = dnn.get_graph().get_operations();
            let mut stepped_reprs: Vec<RepresentationId> =
                dnn.get_input_representation_ids().clone();

            for op_node in operation_nodes.iter() {
                if let Some(num_steps) = op_node.get_operation().num_steps() {
                    for step in 0..(num_steps - 1) {
                        // Avoids underflow
                        if step == num_steps - 1 {
                            continue;
                        }

                        stepped_reprs
                            .push(op_node.get_output_ids()[0].clone().with_step(Some(step)));
                    }
                }
                stepped_reprs.push(op_node.get_output_ids()[0].clone());
            }

            // Iterate over all pairs of stepped_reprs
            let stepped_pairs: Vec<_> = stepped_reprs.into_iter().combinations(2).collect();

            (Just(dnn), select(stepped_pairs))
        })
    }

    fn fcdnn_with_local_output_bounds(
        input_size: usize,
        output_size: usize,
        n_hidden_layers: usize,
        max_layer_width: usize,
    ) -> impl Strategy<Value = (DNN, RepresentationId, RepresentationId, Bounds1)> {
        let strat =
            fcdnn_with_star_end_repr(input_size, output_size, n_hidden_layers, max_layer_width);
        let strat = Strategy::prop_flat_map(strat, move |(dnn, repr_ids)| {
            let start_id = repr_ids[0];
            let end_id = repr_ids[1];

            let bounds_dims = if dnn.get_input_representation_ids().contains(&start_id) {
                let input_op_id = dnn.get_graph().get_representation_input_op_ids(&start_id)[0];
                let op_node = dnn.get_graph().get_operation_node(&input_op_id).unwrap();
                op_node.get_operation().inputs_dims()[0]
            } else {
                let op_id = dnn.get_graph().get_representation_op_id(&start_id).unwrap();
                let op_node = dnn.get_graph().get_operation_node(&op_id).unwrap();
                op_node.get_operation().outputs_dims()[0]
            };

            (Just(dnn), Just(start_id), Just(end_id), Just(bounds_dims))
        });
        Strategy::prop_flat_map(strat, move |(dnn, start_id, end_id, ndims)| {
            (Just(dnn), Just(start_id), Just(end_id), bounds1(ndims))
        })
    }

    proptest! {
        /// Tests whether deep poly runs without failure
        #[test]
        fn test_deeppoly_with_dnn(dnn in fc_dnn(2, 2, 3, 2), local_output_bounds in bounds1(2)) {
            let input_representations = vec![(dnn.get_input_representation_ids()[0], &local_output_bounds)];
            let output_ids = dnn.get_output_representation_ids();
            deep_poly(&dnn, &input_representations, output_ids);
        }

        #[test]
        fn test_fcdnn_with_local_output_bounds((_, start_repr, end_repr, _) in fcdnn_with_local_output_bounds(2,2,3,2)) {
            prop_assert!(start_repr != end_repr);
            if start_repr.representation_node_id == end_repr.representation_node_id {
                prop_assert!(end_repr.operation_step == None || end_repr.operation_step > start_repr.operation_step);
            } else {
                prop_assert!(end_repr.representation_node_id > start_repr.representation_node_id);
            }
        }

        /// Tests whether the abstract bounds are tigheter than the concrete bounds. This is tested by checking
        /// if the abstract bounds concretized with the input bounds are contained by the concrete bounds.
        #[test]
        fn test_abstract_bounds_tighter_than_concrete((dnn, start_repr, end_repr, input_bounds) in fcdnn_with_local_output_bounds(4,4,4,4)) {
            let deep_poly_inputs = vec![(start_repr, &input_bounds)];
            let (concretized_bounds, concrete_bounds) = _deep_poly(&dnn, &deep_poly_inputs, &vec![end_repr]);
            prop_assert!(concretized_bounds[0].is_subset_of(&concrete_bounds[0]), "Concrete bounds {:?} does not contain concretized bounds: {:?}", concrete_bounds, concretized_bounds);
        }

        /// Tests whether the concrete output bounds of deep poly over-approximate, i.e. given any input in
        /// the input bounds you cannot reach a value outside of the concrete bounds.
        #[test]
        fn test_bounds_over_approximate(dnn in fc_dnn(2,2,2,2), input_bounds in bounds1(2)) {
            // Create a starset
            let input_star = Star2::default(&TensorShape::new(vec![Some(input_bounds.ndim())]));
            let starset = GraphStarset::new(dnn, input_bounds.clone(), input_star);

            // Run starsets forward
            let engine = Engine::new(starset.get_graph());
            let inputs = vec![(
                starset.get_dnn().get_input_representation_ids()[0].clone(),
                vec![starset.get_root_id()],
            )];
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
            prop_assert!(res.is_ok());

            let stepped_reprs = get_all_stepped_repr_ids(starset.get_dnn());
            let stepped_pairs: Vec<_> = stepped_reprs.into_iter().combinations(2).collect();

            for repr_pair in stepped_pairs.into_iter() {
                let start_repr = repr_pair[0];
                let end_repr = repr_pair[1];

                let out_dims = {
                    let out_op_id = starset.get_dnn().get_graph().get_representation_op_id(&end_repr).unwrap();
                    starset.get_dnn().get_graph().get_operation_node(&out_op_id).unwrap().get_operation().outputs_dims()[0]
                };

                // Iterate over all corresponding stars
                for star_id in starset.get_stars_for_representation(&start_repr).into_iter() {
                    let star = starset.get_star(star_id);

                    let parent_local_output_bounds = {
                        if let Some(producing_rel) = starset.get_producing_relationship(&star_id) {
                            prop_assert_eq!(producing_rel.input_star_ids.len(), 1);
                            let parent_star_id = producing_rel.input_star_ids[0];
                            let local_output_bounds_opt = starset.get_local_output_bounds(parent_star_id);
                            if local_output_bounds_opt.is_some() {
                                local_output_bounds_opt.as_ref().unwrap().clone()
                            } else {
                                star.calculate_output_axis_aligned_bounding_box(&input_bounds)
                            }
                        } else {
                            star.calculate_output_axis_aligned_bounding_box(&input_bounds)
                        }
                    };

                    // Run DeepPoly for the star until the end representation
                    let deep_poly_inputs = vec![(start_repr, &parent_local_output_bounds)];
                    let concretized_bounds = deep_poly(&starset.get_dnn(), &deep_poly_inputs, &vec![end_repr]);
                    prop_assert_eq!(concretized_bounds.len(), 1);

                    // Check that the output bounds of each leaf star is contained by the concretized deep poly bounds.
                    let test_bounds = {
                        let (bounds_lower, bounds_upper): (Vec<NNVFloat>, Vec<NNVFloat>) =
                            (0..out_dims).map(|dim| {
                                let output_min = star.get_output_min(dim, &input_bounds);
                                let output_max = star.get_output_min(dim, &input_bounds);
                                (output_min, output_max)
                            }).unzip();
                        let bounds_lower = Array1::from_vec(bounds_lower);
                        let bounds_upper = Array1::from_vec(bounds_upper);
                        Bounds1::new(bounds_lower.view(), bounds_upper.view())
                    };

                    prop_assert_eq!(concretized_bounds[0].ndim(), test_bounds.ndim());
                    prop_assert!(test_bounds.is_subset_of(&concretized_bounds[0]), "Deep Poly Bounds: {:?} Test Bounds: {:?}", &concretized_bounds, &test_bounds);
                }
            }
        }
    }
}

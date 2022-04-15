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
    let outputs = engine
        .run_nodal(
            output_ids,
            &input_representations,
            |op_id: OperationId,
             op_node: &OperationNode,
             inputs: &Vec<&(Bounds1, Affine2, Affine2)>,
             op_step: Option<usize>|
             -> (Option<usize>, Vec<(Bounds1, Affine2, Affine2)>) {
                // op_step is None if nothing has run yet, output None as the step when the entire Op is done
                // This visitor, if a step is taken from None, should increment None -> 0 and then op.num_steps -> None
                let (bounds_concrete, laff, uaff): (Vec<_>, Vec<_>, Vec<_>) = inputs
                    .into_iter()
                    .map(|&tup| (&tup.0, &tup.1, &tup.2))
                    .multiunzip();
                let output_id = output_ids
                    .iter()
                    .filter(|&out_id| out_id.operation_step.is_some())
                    .find_or_first(|&repr_id| {
                        op_node
                            .get_output_ids()
                            .iter()
                            .find_or_first(|op_repr_id| {
                                op_repr_id.representation_node_id == repr_id.representation_node_id
                            })
                            .is_some()
                    });
                if op_step.is_some() || output_id.is_some() {
                    let (repr_step, next_step) =
                        get_next_step(op_step, op_node.get_operation().num_steps());
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
        )
        .unwrap();

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
    use std::collections::HashMap;

    use super::*;
    use crate::dnn::dense::Dense;
    use crate::dnn::dnn::DNN;
    use crate::dnn::relu::ReLU;
    use crate::graph::PhysicalOp;
    use crate::star::Star2;
    use crate::test_util::*;
    use proptest::prelude::*;

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

    proptest! {
        /// Tests whether deep poly runs without failure
        #[test]
        fn test_deeppoly_with_dnn(dnn in fc_dnn(2, 2, 3, 2), local_output_bounds in bounds1(2)) {
            let input_representations = vec![(dnn.get_input_representation_ids()[0], &local_output_bounds)];
            let output_ids = dnn.get_output_representation_ids();
            deep_poly(&dnn, &input_representations, output_ids);
        }

        /// Tests whether the abstract bounds are tigheter than the concrete bounds. This is tested by checking
        /// if the abstract bounds concretized with the input bounds are contained by the concrete bounds.
        #[test]
        fn test_abstract_bounds_tighter_than_concrete(dnn in fc_dnn(2,2,3,2), (input_star, input_bounds) in generic_non_empty_star_with_bounds(2, 0)) {
            let engine = Engine::new(dnn.get_graph());
            let star_input_bounds = input_star.calculate_output_axis_aligned_bounding_box(&input_bounds);

            let local_output_bounds: HashMap<RepresentationId, Option<Bounds1>> = HashMap::new();
            local_output_bounds.insert(dnn.get_input_representation_ids()[0].clone(), Some(input_bounds.clone()));

            let inputs: Vec<(RepresentationId, ())> = vec![(dnn.get_input_representation_ids()[0], ())];
            let _outputs = engine.run_nodal(dnn.get_output_representation_ids(), &inputs, |_, op_node: &OperationNode, inputs: &Vec<&()>, step| -> (Option<usize>, Vec<()>) {
                assert_eq!(1, inputs.len());
                let (repr_step, next_step) = get_next_step(op_node.get_operation().num_steps(), step);
                let out_repr_id = op_node.get_output_ids()[0].clone().with_step(repr_step);

                let parent_local_output_bounds = (if let Some(step) = next_step {
                    local_output_bounds.get(&out_repr_id).unwrap().as_ref()
                } else {
                    local_output_bounds.get(&op_node.get_input_ids()[0]).unwrap().as_ref()
                }).map(|bounds| vec![bounds]);

                let output_stars_bounds_opts = op_node.get_operation().forward_star(vec![&input_star], next_step, &star_input_bounds, parent_local_output_bounds);
                assert_eq!(1, output_stars_bounds_opts.len());
                output_stars_bounds_opts[0].into_iter()
                    .map(|(star, local_output_bounds_opt)| {
                        if let Some(local_output_bounds) = local_output_bounds_opt {
                            (star, local_output_bounds)
                        } else {
                            (star, op_node.get_operation())
                        }
                    }).for_each(|(_, local_output_bounds)| {
                        let deep_poly_inputs = vec![(out_repr_id, local_output_bounds)];
                        let (concretized_bounds, concrete_bounds) = _deep_poly(&dnn, &deep_poly_inputs, dnn.get_output_representation_ids());
                        assert!(concretized_bounds[0].is_subset_of(&concrete_bounds[0]), "Concrete bounds {:?} does not contain concretized bounds: {:?}", concrete_bounds, concretized_bounds);
                    });

                (repr_step, vec![()])




                // let deep_poly_inputs = vec![(op_node.get_input_ids()[0].clone(), &inputs[0].1)];
                // let parent_star = vec![&inputs[0].0];
                // let star_outputs = op_node.get_operation().forward_star(parent_star, next_step, &input_bounds, Some(vec![&inputs[0].1]));

                // star_outputs.iter().map(|)
                // let local_output_bounds = star_outputs.1[0].unwrap();

                // let (concretized_bounds, concrete_bounds) = _deep_poly(
                //     &dnn,
                //     &deep_poly_inputs,
                //     dnn.get_output_representation_ids()
                // );
                // assert!(concretized_bounds[0].is_subset_of(&concrete_bounds[0]), "Concrete bounds {:?} does not contain concretized bounds: {:?}", concrete_bounds, concretized_bounds);
                // (repr_step, (local_output_bounds))
            });
        }

        /// Tests whether the concrete output bounds of deep poly over-approximate, i.e. given any input in
        /// the input bounds you cannot reach a value outside of the concrete bounds.
        #[test]
        fn test_bounds_over_approximate(dnn in fc_dnn(2,2,3,2), input_bounds in bounds1(2)) {

        }
    }
}

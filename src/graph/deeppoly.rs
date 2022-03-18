use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::dnn::DNN;
use crate::graph::{Engine, Operation, RepresentationId};
use itertools::MultiUnzip;
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
pub fn deep_poly(dnn: &DNN, input_nodes: Vec<(RepresentationId, Bounds1)>) -> Vec<Bounds1> {
    assert!(!input_nodes.is_empty());
    let input_representations: Vec<(RepresentationId, (Bounds1, Affine2, Affine2))> = {
        // Calculate total input size
        let cum_size: Vec<usize> = input_nodes
            .iter()
            .scan(0, |state, &(_, bounds)| {
                *state += bounds.ndim();
                Some(*state)
            })
            .collect();
        let input_size: usize = input_nodes.iter().map(|(_, bounds)| bounds.ndim()).sum();

        let all_lower_matrix = Array2::eye(input_size); // input_size x input_size
        let all_upper_matrix = Array2::eye(input_size); // input_size x input_size

        input_nodes
            .into_iter()
            .scan(0, |start_idx, (id, bounds)| {
                let dim = bounds.ndim();
                let end_idx: usize = *start_idx + dim;
                let output = (
                    id,
                    (
                        bounds,
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

    // op_step is None if nothing has run yet, output None as the step when the entire Op is done
    // This visitor, if a step is taken from None, should increment None -> 0 and then op.num_steps -> None
    let visitor = |op: &dyn Operation,
                   inputs: Vec<&(Bounds1, Affine2, Affine2)>,
                   op_step: Option<usize>|
     -> (Option<usize>, Vec<(Bounds1, Affine2, Affine2)>) {
        let (bounds_concrete, laff, uaff) = inputs.iter().map(|&x| *x).multiunzip();
        if let Some(step) = op_step {
            let new_step = if (step + 1) == (op.num_steps().unwrap() - 1) {
                None
            } else {
                Some(step + 1)
            };
            (
                new_step,
                op.apply_bounds_step(step, &bounds_concrete, &laff, &uaff),
            )
        } else {
            (None, op.apply_bounds(&bounds_concrete, &laff, &uaff))
        }
    };

    let engine = Engine::new(dnn.get_graph());

    let outputs = engine
        .run(
            dnn.get_output_representation_ids().clone(),
            input_representations,
            visitor,
        )
        .unwrap();

    // Collect all input bounds into one bound
    let input_bounds = input_nodes
        .iter()
        .fold(Bounds1::default(0), |acc, &(_, b)| acc.append(&b));

    // Concretize the bounds in terms of the input bounds
    outputs
        .into_iter()
        .map(|(_, (_, lower_aff, upper_aff))| {
            let lower_bounds = lower_aff.signed_apply(&input_bounds);
            let upper_bounds = upper_aff.signed_apply(&input_bounds);
            let bounds = Bounds1::new(lower_bounds.lower(), upper_bounds.upper());
            debug_assert!(bounds.bounds_iter().into_iter().all(|x| x[[0]] <= x[[1]]));
            bounds
        })
        .collect::<Vec<_>>()
}

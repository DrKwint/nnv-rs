use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::dnn::dnn::DNN;
use crate::graph::Engine;
use crate::graph::RepresentationId;
use crate::NNVFloat;
use log::trace;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::Zip;
use num::Float;
use num::Zero;
use std::ops::Neg;

/// # Panics
pub fn deep_poly_steprelu(
    dim: usize,
    mut bounds: Bounds1,
    mut lower_aff: Affine2,
    mut upper_aff: Affine2,
) -> (Bounds1, (Affine2, Affine2)) {
    let mut bounds_slice = bounds.index_mut(dim);
    let (mut lbasis, mut lshift) = lower_aff.get_eqn_mut(dim);
    let (mut u_basis, mut u_shift) = upper_aff.get_eqn_mut(dim);
    let l = bounds_slice[[0]];
    let u = bounds_slice[[1]];
    if u <= NNVFloat::zero() {
        bounds_slice.fill(NNVFloat::zero());
        lbasis.fill(NNVFloat::zero());
        u_basis.fill(NNVFloat::zero());
        lshift.fill(NNVFloat::zero());
        u_shift.fill(NNVFloat::zero());
    // gt branch
    } else if l >= NNVFloat::zero() {
        // here, leave mul and shift at defaults
        // then, spanning branch
    } else {
        // Using y = ax + b:
        // Handling so things don't break down in the infinite case
        if u == NNVFloat::infinity() {
            u_basis.mapv_inplace(|x| {
                if x * NNVFloat::infinity() == NNVFloat::nan() {
                    0.
                } else {
                    NNVFloat::INFINITY
                }
            });
            u_shift.mapv_inplace(|x| {
                if x * NNVFloat::infinity() == NNVFloat::nan() {
                    0.
                } else {
                    NNVFloat::INFINITY
                }
            });
        } else {
            u_basis.mapv_inplace(|a| a * (u / (u - l)));
            u_shift.mapv_inplace(|b| u * (b - l) / (u - l));
        }

        // use approximation with least area
        if u < NNVFloat::neg(l) || l == NNVFloat::neg_infinity() {
            // Eqn. 3 from the paper
            *bounds_slice.get_mut(0).unwrap() = NNVFloat::zero();
            lbasis.fill(NNVFloat::zero());
            lshift.fill(NNVFloat::zero());
        } else {
            // Eqn. 4 from the paper, leave l_mul at default
        }
    }
    //debug_assert!(bounds.is_all_finite());
    (bounds, (lower_aff, upper_aff))
}

pub fn deep_poly_relu(
    bounds: &Bounds1,
    lower_aff: &Affine2,
    upper_aff: &Affine2,
) -> (Bounds1, (Affine2, Affine2)) {
    let mut out = bounds.clone();
    let mut l_mul = Array1::ones(bounds.ndim());
    let mut u_mul = Array1::ones(bounds.ndim());
    let mut u_shift = Array1::zeros(bounds.ndim());
    Zip::from(bounds.bounds_iter())
        .and(out.bounds_iter_mut())
        .and(&mut l_mul)
        .and(&mut u_mul)
        .and(&mut u_shift)
        .for_each(
            |b: ArrayView1<NNVFloat>,
             mut out: ArrayViewMut1<NNVFloat>,
             l_mul: &mut NNVFloat,
             u_mul: &mut NNVFloat,
             u_shift: &mut NNVFloat| {
                let l = b[0];
                let u = b[1];
                // lt branch
                if u <= NNVFloat::zero() {
                    out[0] = NNVFloat::zero();
                    out[1] = NNVFloat::zero();
                    *l_mul = NNVFloat::zero();
                    *u_mul = NNVFloat::zero();
                // gt branch
                } else if l >= NNVFloat::zero() {
                    // Leave mul and shift at defaults
                    // spanning branch
                } else {
                    *u_mul = u / (u - l);
                    *u_shift = NNVFloat::neg((u * l) / (u - l));
                    // use approximation with least area
                    if u < NNVFloat::neg(l) {
                        // Eqn. 3 from the paper
                        out[0] = NNVFloat::zero();
                        *l_mul = NNVFloat::zero();
                    } else {
                        // Eqn. 4 from the paper, leave l_mul at default
                    }
                }
            },
        );
    let mut lower_aff = lower_aff.clone();
    lower_aff.scale_eqns(l_mul.view());
    let mut upper_aff = upper_aff.clone();
    upper_aff.scale_eqns(u_mul.view());
    upper_aff = upper_aff + u_shift;
    (out, (lower_aff, upper_aff))
}

/// Runs deep poly on a network starting from a set of given starting nodes
///
/// # Description
///
/// Approximates output bounds on of a network given input bounds for specific inputs.
///
/// # Arguments
///
/// * `input_bounds` - The input bounds for each node in `input_nodes`, i.e., each input to the sub-network.
/// * `dnn` - A reference to the original dnn
/// * `input_nodes` - The id of the representation of the dnn that each `input_bounds`
///     entry corresponds to. As such, the size of the vectors should be the same.
///
/// # Panics
pub fn deep_poly(
    input_bounds: &Vec<Bounds1>,
    dnn: &DNN,
    input_nodes: Vec<(RepresentationId, Option<usize>, Vec<Bounds1>)>,
) -> Vec<Bounds1> {
    debug_assert_eq!(input_bounds.len(), input_nodes.len());
    trace!("with input bounds {:?}", input_bounds);
    debug_assert!(
        input_bounds
            .iter()
            .map(|bounds| bounds.bounds_iter().into_iter().all(|x| (x[[0]] <= x[[1]])))
            .all(|x| x),
        "Input bounds are flipped!"
    );
    let inputs = input_bounds
        .iter()
        .zip(dnn.get_input_representation_ids().iter())
        .map(|(&bounds, &id)| {
            let ndim = bounds.ndim();
            (
                id,
                (
                    bounds.clone(),
                    (Affine2::identity(ndim), Affine2::identity(ndim)),
                ),
            )
        })
        .collect::<Vec<_>>();

    let engine = Engine::new(dnn.get_graph());

    let visit = |op: &Box<dyn crate::graph::Operation>,
                 inputs: Vec<&(Bounds1, (Affine2, Affine2))>,
                 step: Option<usize>|
     -> Vec<(Bounds1, (Affine2, Affine2))> {
        let (bounds_concrete, (laff, uaff)) = inputs.iter().map(|&x| *x).unzip();

        let out = if let Some(dim) = step {
            op.apply_bounds_step(dim, &bounds_concrete, &laff, &uaff)
        } else {
            op.apply_bounds(&bounds_concrete, &laff, &uaff)
        };
        debug_assert!(
            out.iter().all(|o| o
                .0
                .bounds_iter()
                .into_iter()
                .all(|x| (x[[0]] <= x[[1]]) || (x[[0]] - x[[1]]) < 1e-4)),
            "Bounds: {:?}",
            out.iter().map(|x| x.0).collect::<Vec<_>>()
        );
        if cfg!(debug_assertions) {
            for o in out {
                let lower_bounds = o.1 .0.signed_apply(&input_bounds[0]);
                let upper_bounds = o.1 .1.signed_apply(&input_bounds[0]);
                let realized_abstract_bounds =
                    Bounds1::new(lower_bounds.lower(), upper_bounds.upper());
                debug_assert!(
                    realized_abstract_bounds.subset(&o.0),
                    "\n\nRealized abstract: {:?}\nConcrete: {:?}\n\n",
                    realized_abstract_bounds,
                    bounds_concrete
                );
            }
        }
        out
    };

    let aff_bounds = engine.run(
        dnn.get_output_representation_ids().clone(),
        inputs,
        true,
        visit,
    );

    // Final substitution to get output bounds
    let bounds = aff_bounds
        .unwrap()
        .into_iter()
        .map(|(conc_bounds, (l_bounds, u_bounds))| -> Bounds1 {
            todo!();
            // let lower_bounds = l_bounds.signed_apply(input_bounds);
            // let upper_bounds = u_bounds.signed_apply(input_bounds);
            // let bounds = Bounds1::new(lower_bounds.lower(), upper_bounds.upper());
            // debug_assert!(bounds.bounds_iter().into_iter().all(|x| x[[0]] <= x[[1]]));
            // bounds
        })
        .collect::<Vec<_>>();
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::dense::Dense;
    use crate::dnn::dnn::DNN;
    use crate::dnn::relu::ReLU;
    use crate::test_util::{bounds1, fc_dnn};
    use ndarray::Array2;
    use ndarray::Axis;
    use proptest::proptest;

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
        let _dnn = DNN::from_sequential(vec![
            Box::new(dense1),
            Box::new(relu1),
            Box::new(dense2),
            Box::new(relu2),
        ]);
        let _bounds: Bounds1 = Bounds1::new(
            Array1::from_vec(vec![0.0, 0.0, 0.]).view(),
            Array1::zeros(3).view(),
        );
    }

    proptest! {
        #[test]
        fn test_deeppoly_with_dnn(dnn in fc_dnn(2, 2, 1, 2), input_bounds in bounds1(2)) {
            deep_poly(&input_bounds, &dnn, DNNIterator::new(&dnn, DNNIndex::default()));
        }
    }

    #[test]
    fn test_deeppoly_relu_gt_correctness() {
        let bounds: Bounds1 = Bounds1::new(Array1::zeros(4).view(), Array1::ones(4).view());
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, bounds);
        assert_eq!(new_l, lower_aff);
        assert_eq!(new_u, upper_aff);
    }

    #[test]
    fn test_deeppoly_relu_lt_correctness() {
        let bounds: Bounds1 = Bounds1::new((Array1::ones(4) * -1.).view(), Array1::zeros(4).view());
        let lower_aff = Affine2::identity(4) + (-4.);
        let upper_aff = Affine2::identity(4);
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(
            new_b,
            Bounds1::new(Array1::zeros(4).view(), Array1::zeros(4).view())
        );
        assert_eq!(new_l, Affine2::identity(4) * 0.);
        assert_eq!(new_u, Affine2::identity(4) * 0.);
    }

    #[test]
    fn test_deeppoly_relu_spanning_firstbranch_correctness() {
        let bounds: Bounds1 = Bounds1::new((Array1::ones(4) * -2.).view(), Array1::ones(4).view());
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let upper_aff_update = Affine2::new(
            Array2::from_diag(&(&bounds.upper() / (&bounds.upper() - &bounds.lower()))),
            &bounds.upper() * &bounds.lower() / (&bounds.upper() - &bounds.lower()) * -1.,
        );
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(
            new_b,
            Bounds1::new(Array1::zeros(4).view(), Array1::ones(4).view())
        );
        assert_eq!(new_l, lower_aff * 0.);
        assert_eq!(new_u, upper_aff * &upper_aff_update);
    }

    #[test]
    fn test_deeppoly_relu_spanning_secondbranch_correctness() {
        let bounds: Bounds1 = Bounds1::new((Array1::ones(4) * -1.).view(), Array1::ones(4).view());
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let upper_aff_update = Affine2::new(
            Array2::from_diag(&(&bounds.upper() / (&bounds.upper() - &bounds.lower()))),
            &bounds.upper() * &bounds.lower() / (&bounds.upper() - &bounds.lower()) * -1.,
        );
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, bounds);
        assert_eq!(new_l, lower_aff);
        assert_eq!(new_u, upper_aff * &upper_aff_update);
    }
}

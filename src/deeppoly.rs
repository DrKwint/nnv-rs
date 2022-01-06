use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::dnn::DNNIterator;
use crate::NNVFloat;
use log::{debug, trace};
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::Zip;
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
        u_basis.mapv_inplace(|a| a * (u / (u - l)));
        u_shift.mapv_inplace(|b| u * (b - l) / (u - l));

        // use approximation with least area
        if u < NNVFloat::neg(l) {
            // Eqn. 3 from the paper
            *bounds_slice.get_mut(0).unwrap() = NNVFloat::zero();
            lbasis.fill(NNVFloat::zero());
            lshift.fill(NNVFloat::zero());
        } else {
            // Eqn. 4 from the paper, leave l_mul at default
        }
    }
    debug_assert!(bounds.is_all_finite());
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

/// # Panics
pub fn deep_poly(input_bounds: &Bounds1, dnn_iter: DNNIterator) -> Bounds1 {
    debug!("Starting Deeppoly at {:?}", dnn_iter.get_idx());
    trace!("with input bounds {:?}", input_bounds);
    debug_assert!(
        input_bounds
            .bounds_iter()
            .into_iter()
            .all(|x| (x[[0]] <= x[[1]])),
        "Input bounds are flipped!"
    );
    let ndim = input_bounds.ndim();
    // Affine expressing bounds on each variable in current layer as a
    // linear function of input bounds
    let aff_bounds = dnn_iter.fold(
        // Initialize with identity
        (
            input_bounds.clone(),
            (Affine2::identity(ndim), Affine2::identity(ndim)),
        ),
        |(bounds_concrete, (laff, uaff)), op| {
            let out = op.apply_bounds(&bounds_concrete, &laff, &uaff);
            debug_assert!(
                out.0
                    .bounds_iter()
                    .into_iter()
                    .all(|x| (x[[0]] <= x[[1]]) || (x[[0]] - x[[1]]) < 1e-4),
                "Bounds: {:?}",
                out.0
            );
            if cfg!(debug_assertions) {
                let lower_bounds = out.1 .0.signed_apply(input_bounds);
                let upper_bounds = out.1 .1.signed_apply(input_bounds);
                let realized_abstract_bounds =
                    Bounds1::new(lower_bounds.lower(), upper_bounds.upper());
                debug_assert!(
                    realized_abstract_bounds.subset(&out.0),
                    "\n\nRealized abstract: {:?}\nConcrete: {:?}\n\n",
                    realized_abstract_bounds,
                    bounds_concrete
                );
            }
            out
        },
    );
    // Final substitution to get output bounds
    let lower_bounds = aff_bounds.1 .0.signed_apply(input_bounds);
    let upper_bounds = aff_bounds.1 .1.signed_apply(input_bounds);
    let bounds = Bounds1::new(lower_bounds.lower(), upper_bounds.upper());
    debug_assert!(bounds.bounds_iter().into_iter().all(|x| x[[0]] <= x[[1]]));
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::{DNNIndex, Layer, DNN};
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
        let dense1 = Layer::new_dense(aff1);
        let relu1: Layer = Layer::new_relu(1);
        let aff2 = Affine2::new(
            Array1::from_vec(vec![9.49, 0.0]).insert_axis(Axis(1)),
            Array1::from_vec(vec![0., 0.]),
        );
        let dense2 = Layer::new_dense(aff2);
        let relu2: Layer = Layer::new_relu(2);
        let _dnn = DNN::new(vec![dense1, relu1, dense2, relu2]);
        let _bounds: Bounds1 = Bounds1::new(
            Array1::from_vec(vec![0.0, 0.0, 0.]).view(),
            Array1::zeros(3).view(),
        );
    }

    proptest! {
        #[test]
        fn test_deeppoly_with_dnn(dnn in fc_dnn(2, 2, 1, 2), input_bounds in bounds1(2)) {
            deep_poly(&input_bounds, DNNIterator::new(&dnn, DNNIndex::default()));
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

use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::dnn::DNNIterator;
use log::{debug, error, log_enabled, trace, warn, Level};
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::ScalarOperand;
use ndarray::Zip;
use num::Float;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::MulAssign;

pub fn deep_poly_steprelu<T: 'static + Float + Default>(
    dim: usize,
    mut bounds: Bounds1<T>,
    mut lower_aff: Affine2<T>,
    mut upper_aff: Affine2<T>,
) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
    let mut bounds_slice = bounds.index_mut(dim);
    let (mut lbasis, mut lshift) = lower_aff.get_eqn_mut(dim);
    let (mut ubasis, mut ushift) = upper_aff.get_eqn_mut(dim);
    let l = bounds_slice[[0]];
    let u = bounds_slice[[1]];
    if u <= T::zero() {
        bounds_slice.fill(T::zero());
        lbasis.fill(T::zero());
        ubasis.fill(T::zero());
        lshift.fill(T::zero());
        ushift.fill(T::zero());
    // gt branch
    } else if l >= T::zero() {
        // here, leave mul and shift at defaults
        // then, spanning branch
    } else {
        ubasis.mapv_inplace(|x| x * (u / (u - l)));
        ushift.mapv_inplace(|x| x - ((u * l) / (u - l)));
        // use approximation with least area
        if u < T::neg(l) {
            // Eqn. 3 from the paper
            *bounds_slice.get_mut(0).unwrap() = T::zero();
            lbasis.fill(T::zero());
            lshift.fill(T::zero());
        } else {
            // Eqn. 4 from the paper, leave l_mul at default
        }
    }
    debug_assert!(bounds.is_all_finite());
    (bounds, (lower_aff, upper_aff))
}

pub fn deep_poly_relu<
    T: Float + Default + std::ops::MulAssign + ScalarOperand + std::ops::Mul + std::fmt::Debug,
>(
    bounds: &Bounds1<T>,
    lower_aff: &Affine2<T>,
    upper_aff: &Affine2<T>,
) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
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
            |b: ArrayView1<T>,
             mut out: ArrayViewMut1<T>,
             l_mul: &mut T,
             u_mul: &mut T,
             u_shift: &mut T| {
                let l = b[0];
                let u = b[1];
                // lt branch
                if u <= T::zero() {
                    out[0] = T::zero();
                    out[1] = T::zero();
                    *l_mul = T::zero();
                    *u_mul = T::zero();
                // gt branch
                } else if l >= T::zero() {
                    // Leave mul and shift at defaults
                    // spanning branch
                } else {
                    *u_mul = u / (u - l);
                    *u_shift = T::neg((u * l) / (u - l));
                    // use approximation with least area
                    if u < T::neg(l) {
                        // Eqn. 3 from the paper
                        out[0] = T::zero();
                        *l_mul = T::zero();
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

pub fn deep_poly<T: 'static + Float>(
    input_bounds: Bounds1<T>,
    dnn_iter: DNNIterator<T>,
) -> Bounds1<T>
where
    T: ScalarOperand + Display + Debug + Default + MulAssign + std::convert::From<f64> + Sum,
    f64: std::convert::From<T>,
{
    debug!("Starting Deeppoly at {:?}", dnn_iter.get_idx());
    trace!("with input bounds {:?}", input_bounds);
    let ndim = input_bounds.ndim();
    // Affine expressing bounds on each variable in current layer as a
    // linear function of input bounds
    let aff_bounds = dnn_iter.fold(
        // Initialize with identity
        //(Affine2::identity(ndim), Affine2::identity(ndim)),
        (
            Affine2::new(
                Array2::from_diag(&input_bounds.lower()),
                Array1::zeros(ndim),
            ),
            Affine2::new(
                Array2::from_diag(&input_bounds.upper()),
                Array1::zeros(ndim),
            ),
        ),
        |(laff, uaff), op| {
            trace!("op {:?}", op);
            // Substitute input concrete bounds into current abstract bounds
            // to get current concrete bounds
            let bounds_concrete = Bounds1::new(
                laff.apply(&Array1::ones(ndim).view()),
                uaff.apply(&Array1::ones(ndim).view()),
            );
            let out = op.apply_bounds(bounds_concrete, laff, uaff);
            trace!("new bounds {:?}", out.0);
            out.1
        },
    );
    // Final substitution to get output bounds
    Bounds1::new(
        aff_bounds.0.apply(&Array1::ones(ndim).view()),
        aff_bounds.1.apply(&Array1::ones(ndim).view()),
    )
    //aff_bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::{DNNIndex, Layer, DNN};
    use crate::test_util::{bounds1, fc_dnn};
    use ndarray::Array2;
    use ndarray::Axis;
    use ndarray::Ix1;
    use proptest::{prop_assert, proptest};

    #[test]
    fn test_deeppoly_concrete() {
        let aff1: Affine2<f64> = Affine2::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]).insert_axis(Axis(0)),
            Array1::from_vec(vec![7.85]),
        );
        let dense1 = Layer::new_dense(aff1);
        let relu1: Layer<f64> = Layer::new_relu(1);
        let aff2 = Affine2::new(
            Array1::from_vec(vec![9.49, 0.0]).insert_axis(Axis(1)),
            Array1::from_vec(vec![0., 0.]),
        );
        let dense2 = Layer::new_dense(aff2);
        let relu2: Layer<f64> = Layer::new_relu(2);
        let dnn = DNN::new(vec![dense1, relu1, dense2, relu2]);
        let bounds: Bounds1<f64> =
            Bounds1::new(Array1::from_vec(vec![0.0, 0.0, 0.]), Array1::zeros(3));
    }

    proptest! {
        #[test]
        fn test_deeppoly_with_dnn(dnn in fc_dnn(2, 2, 1, 2), input_bounds in bounds1(2)) {
            deep_poly(input_bounds, DNNIterator::new(&dnn, DNNIndex::default()));
        }
    }

    proptest! {
        #[test]
        fn test_deeppoly_correctness(dnn in fc_dnn(8, 4, 5, 5), input_bounds in bounds1(8)) {
            let concrete_input = input_bounds.sample_uniform(0u64);
            let output_bounds = deep_poly(input_bounds, DNNIterator::new(&dnn, DNNIndex::default()));
            let concrete_output = dnn.forward(concrete_input.into_dyn()).into_dimensionality::<Ix1>().unwrap();
            prop_assert!(output_bounds.is_member(&concrete_output.view()), "\n\nConcrete output: {}\nOutput bounds: {}\n\n", concrete_output, output_bounds)
        }
    }

    #[test]
    fn test_deeppoly_relu_gt_correctness() {
        let bounds: Bounds1<f64> = Bounds1::new(Array1::zeros(4), Array1::ones(4));
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, bounds);
        assert_eq!(new_l, lower_aff);
        assert_eq!(new_u, upper_aff);
    }

    #[test]
    fn test_deeppoly_relu_lt_correctness() {
        let bounds: Bounds1<f64> = Bounds1::new(Array1::ones(4) * -1., Array1::zeros(4));
        let lower_aff = Affine2::identity(4) + (-4.);
        let upper_aff = Affine2::identity(4);
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, Bounds1::new(Array1::zeros(4), Array1::zeros(4)));
        assert_eq!(new_l, Affine2::identity(4) * 0.);
        assert_eq!(new_u, Affine2::identity(4) * 0.);
    }

    #[test]
    fn test_deeppoly_relu_spanning_firstbranch_correctness() {
        let bounds: Bounds1<f64> = Bounds1::new(Array1::ones(4) * -2., Array1::ones(4));
        let lower_aff = Affine2::identity(4);
        let upper_aff = Affine2::identity(4);
        let upper_aff_update = Affine2::new(
            Array2::from_diag(&(&bounds.upper() / (&bounds.upper() - &bounds.lower()))),
            &bounds.upper() * &bounds.lower() / (&bounds.upper() - &bounds.lower()) * -1.,
        );
        let (new_b, (new_l, new_u)) = deep_poly_relu(&bounds, &lower_aff, &upper_aff);
        assert_eq!(new_b, Bounds1::new(Array1::zeros(4), Array1::ones(4)));
        assert_eq!(new_l, lower_aff * 0.);
        assert_eq!(new_u, upper_aff * &upper_aff_update);
    }

    #[test]
    fn test_deeppoly_relu_spanning_secondbranch_correctness() {
        let bounds: Bounds1<f64> = Bounds1::new(Array1::ones(4) * -1., Array1::ones(4));
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

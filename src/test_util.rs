#![cfg(test)]
use crate::affine::Affine2;
use crate::bounds::{Bounds, Bounds1};
use crate::constellation::Constellation;
use crate::dnn::Layer;
use crate::dnn::DNN;
use crate::inequality::Inequality;
use crate::polytope::Polytope;
use crate::star::Star2;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Ix2;
use ndarray::Zip;
use proptest::arbitrary::functor::ArbitraryF1;
use proptest::prelude::any;
use proptest::prelude::*;
use proptest::sample::SizeRange;
use std::mem;

prop_compose! {
    pub fn array1(len: usize)
        (v in Vec::lift1_with(-10. .. 10., SizeRange::new(len..=len))) -> Array1<f64> {
            Array1::from_vec(v)
        }
}

prop_compose! {
    pub fn pos_def_array1(len: usize)
        (v in Vec::lift1_with(0.00001 .. 10., SizeRange::new(len..=len))) -> Array1<f64> {
            Array1::from_vec(v)
        }
}

prop_compose! {
    pub fn array2(rows: usize, cols: usize)
        (v in Vec::lift1_with(array1(cols), SizeRange::new(rows..=rows))) -> Array2<f64> {
            assert!(rows > 0);
            ndarray::stack(Axis(0), &v.iter().map(ndarray::ArrayBase::view).collect::<Vec<ArrayView1<f64>>>()).unwrap()
        }
}

prop_compose! {
    pub fn affine2(in_dim: usize, out_dim: usize)
        (basis in array2(out_dim, in_dim), shift in array1(out_dim)) -> Affine2<f64> {
            Affine2::new(basis, shift)
        }
}

prop_compose! {
    pub fn bounds1(len: usize)(mut lower in array1(len), mut upper in array1(len)) -> Bounds1<f64> {
        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| if *l > *u {mem::swap(l, u)});
        assert!(Zip::from(&lower).and(&upper).all(|l, u| l <= u));
        Bounds::new(lower.view(), upper.view())
    }
}

prop_compose! {
    pub fn bounds1_sample(bounds: Bounds1<f64>)(seed in any::<u64>()) -> Array1<f64> {
        bounds.sample_uniform(seed)
    }
}

prop_compose! {
    pub fn fc_dnn(input_size: usize, output_size: usize, nlayers: usize, max_layer_width: usize)
        (repr_sizes in Vec::lift1_with(1..max_layer_width,
                                       SizeRange::new(nlayers..=nlayers))
         .prop_map(move |mut x| {
             x.insert(0, input_size);
             x.push(output_size);
             x
         }))
        (affines in {
            let pairs = repr_sizes.iter()
                .zip(repr_sizes.iter().skip(1));
            pairs.map(|(&x, &y)| affine2(x,y)).collect::<Vec<_>>()}
        ) -> DNN<f64> {
            let mut dnn = DNN::default();
            for aff in affines {
                    let output_dim = aff.output_dim();
                    dnn.add_layer(Layer::new_dense(aff));
                    dnn.add_layer(Layer::new_relu(output_dim));
                }
            dnn
        }
}

prop_compose! {
    pub fn constellation(input_size: usize, output_size: usize, nlayers: usize, max_layer_width: usize)
        (loc in array1(input_size), scale_diag in pos_def_array1(input_size), dnn in fc_dnn(input_size, output_size, nlayers, max_layer_width)) -> Constellation<f64, Ix2>
    {
        let lbs = loc.clone() - 3.5 * scale_diag.clone();
        let ubs = loc.clone() + 3.5 * scale_diag.clone();
        let input_bounds = Bounds1::new(lbs.view(), ubs.view());
        let star = Star2::new(Array2::eye(input_size), Array1::zeros(input_size)).with_input_bounds(input_bounds.clone());
        Constellation::new(star, dnn, Some(input_bounds), loc, Array2::from_diag(&scale_diag).to_owned())
    }
}

prop_compose! {
    pub fn inequality(num_dims: usize, num_constraints: usize)
        (
            mut coeffs in array2(num_constraints, num_dims),
            rhs in array1(num_constraints)
        ) -> Inequality<f64> {
            coeffs.rows_mut().into_iter().for_each(|mut row| {
                if row.iter().all(|x| *x == 0.0_f64) {
                    let mut rng = rand::thread_rng();
                    let one_idx = rng.gen_range(0..row.len());
                    row[one_idx] = 1.0;
                }
            });

            Inequality::new(coeffs, rhs)
        }
}

prop_compose! {
    pub fn inequality_including_zero(num_dims: usize, num_constraints: usize)
        (ineq in inequality(num_dims, num_constraints)) -> Inequality<f64> {
            let zero = Array1::zeros(num_dims);
            let mut inequality = Inequality::new(
                Array2::zeros((0, num_dims)),
                Array1::zeros(0)
            );
            (0..ineq.num_constraints())
                .into_iter()
                .for_each(|eqn_idx| {
                    let eqn = ineq.get_eqn(eqn_idx);
                    if eqn.is_member(&zero.view()) {
                        inequality.add_eqns(&eqn, true);
                    } else {
                        let new_coeffs = -1. * eqn.coeffs().to_owned();
                        let new_rhs = -1. * eqn.rhs().to_owned();
                        let eqn = Inequality::new(new_coeffs, new_rhs);
                        inequality.add_eqns(&eqn, true);
                    }
                });
            inequality
        }
}

prop_compose! {
    pub fn polytope(num_dims: usize, num_constraints: usize)
        (ineq in inequality(num_dims, num_constraints)) -> Polytope<f64> {
            Polytope::from_halfspaces(ineq)
        }
}

prop_compose! {
    pub fn non_empty_polytope(num_dims: usize, num_constraints: usize)
        (
            mut ineq in inequality_including_zero(num_dims, num_constraints)
                .prop_filter("Non-zero intercepts",
                             |i| !i.rhs().iter().any(|x| *x == 0.0_f64))
        ) -> Polytope<f64> {

            // Make a box bigger than possible inner inequalities
            let box_coeffs = Array2::eye(num_dims);
            let mut box_rhs = Array1::ones(num_dims);
            box_rhs *= 20.;

            let upper_box_ineq = Inequality::new(box_coeffs.clone(), box_rhs.clone());
            let lower_box_ineq = Inequality::new(-1. * box_coeffs, box_rhs);

            ineq.add_eqns(&upper_box_ineq, true);
            ineq.add_eqns(&lower_box_ineq, true);
            Polytope::from_halfspaces(ineq)
        }
}

prop_compose! {
    /// Creates an empty polytope
    ///
    /// Constructs an empty polytope by starting with a non-empty
    /// halfspace that includes the origin. It then flips the sign of
    /// the inequality such that the origin is no longer
    /// included. Next, a second inequality is constructed that is
    /// this inequality flipped with respect to the origin (by
    /// multiplying the coefficients by -1. Finally, the inequalities
    /// are appended to each other and used to create a polytope.
    pub fn empty_polytope(num_dims: usize, num_constraints: usize)
        (
            mut ineq in inequality_including_zero(num_dims, num_constraints)
                .prop_filter("Cannot pass through origin",
                             |eq| !eq.rhs().iter().any(|x| *x == 0.0_f64))
        ) -> Polytope<f64> {
            // Invert the sign of the inequality
            ineq *= -1.0;

            // Construct the inequality that is the above flipped
            // across the origin.
            let inverse_ineq = Inequality::new(
                ineq.coeffs().to_owned() * -1.,
                ineq.rhs().to_owned()
            );
            ineq.add_eqns(&inverse_ineq, true);

            // Make a box bigger than possible inner inequalities
            let box_coeffs = Array2::eye(num_dims);
            let mut box_rhs = Array1::ones(num_dims);
            box_rhs *= 20.;

            let upper_box_ineq = Inequality::new(box_coeffs.clone(), box_rhs.clone());
            let lower_box_ineq = Inequality::new(-1. * box_coeffs, box_rhs);

            ineq.add_eqns(&upper_box_ineq, true);
            ineq.add_eqns(&lower_box_ineq, true);

            // Construct the empty polytope.
            Polytope::from_halfspaces(ineq)
        }
}

prop_compose! {
    pub fn non_empty_star(num_dims: usize, num_constraints: usize)
        (
            basis in array2(num_dims, num_dims)
                .prop_filter("Need valid basis",
                             |b| !b.rows().into_iter()
                             .any(|r| r.iter().all(|x| *x == 0.0_f64)) &&
                             !b.columns().into_iter()
                             .any(|r| r.iter().all(|x| *x == 0.0_f64)) &&
                             !b.iter().any(|x| *x == 0.0_f64)),
            center in array1(num_dims),
            constraints in non_empty_polytope(num_dims, num_constraints)
        ) -> Star2<f64> {
            let star = Star2::new(basis, center).with_constraints(constraints);
            assert!(!star.is_empty());
            star
        }
}

prop_compose! {
    pub fn empty_star(num_dims: usize, num_constraints: usize)
        (
            basis in array2(num_dims, num_dims)
                .prop_filter("Need valid basis",
                             |b| !b.rows().into_iter()
                             .any(|r| r.iter().all(|x| *x == 0.0_f64)) &&
                             !b.columns().into_iter()
                             .any(|r| r.iter().all(|x| *x == 0.0_f64)) &&
                             !b.iter().any(|x| *x == 0.0_f64)),
            center in array1(num_dims),
            constraints in empty_polytope(num_dims, num_constraints)
        ) -> Star2<f64> {
            Star2::new(basis, center).with_constraints(constraints)
        }
}

prop_compose! {
    pub fn generic_bounds1(max_len: usize)
        (dim in 1..max_len)
        (bounds in bounds1(dim)) -> Bounds1<f64> {
            bounds
        }
}

prop_compose! {
    pub fn generic_inequality_including_zero(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (ineq in inequality_including_zero(dim, constraints)) -> Inequality<f64> {
            ineq
        }
}

prop_compose! {
    pub fn generic_empty_polytope(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (poly in empty_polytope(dim, constraints)) -> Polytope<f64> {
            poly
        }
}

prop_compose! {
    pub fn generic_non_empty_polytope(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (poly in non_empty_polytope(dim, constraints)) -> Polytope<f64> {
            poly
        }
}

prop_compose! {
    pub fn generic_empty_star(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (star in empty_star(dim, constraints)) -> Star2<f64> {
            star
        }
}

prop_compose! {
    pub fn generic_non_empty_star(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (star in non_empty_star(dim, constraints)) -> Star2<f64> {
            star
        }
}

prop_compose! {
    pub fn generic_fc_dnn(max_input_size: usize, max_output_size: usize, max_nlayers: usize, max_layer_width: usize)
        (input_size in 1..max_input_size, output_size in 1..max_output_size, nlayers in 1..max_nlayers)
        (dnn in fc_dnn(input_size, output_size, nlayers, max_layer_width)) -> DNN<f64>
    {
        dnn
    }
}

prop_compose! {
    pub fn generic_constellation(max_input_size: usize, max_output_size: usize, max_nlayers: usize, max_layer_width: usize)
        (input_size in 1..max_input_size, output_size in 1..max_output_size, nlayers in 1..max_nlayers)
        (constellation in constellation(input_size, output_size, nlayers, max_layer_width)) -> Constellation<f64, Ix2>
        {
            constellation
        }
}

proptest! {
    #[test]
    fn test_inequality_including_zero(ineq in generic_inequality_including_zero(2, 4)) {
        let zero = Array1::zeros(ineq.num_dims());
        prop_assert!(ineq.is_member(&zero.view()));
    }

    #[test]
    fn test_empty_polytope(poly in generic_empty_polytope(2, 4)) {
        prop_assert!(poly.is_empty());
    }

    #[test]
    fn test_non_empty_polytope(poly in generic_non_empty_polytope(2, 4)) {
        prop_assert!(!poly.is_empty());
    }

    #[test]
    fn test_non_empty_star(star in generic_non_empty_star(2, 4)) {
        prop_assert!(!star.is_empty());
    }

    #[test]
    fn test_generic_fc_dnn(_dnn in generic_fc_dnn(5, 5, 5, 5)) {
        // Yes, this is the full test. The test is that we can
        // construct varying sizes of dnns.
    }
}

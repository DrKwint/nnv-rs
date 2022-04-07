#![cfg(test)]
use crate::affine::Affine2;
use crate::bounds::{Bounds, Bounds1};
use crate::dnn::dense::Dense;
use crate::dnn::dnn::DNN;
use crate::dnn::relu::ReLU;
use crate::graph::{Graph, Operation};
use crate::graph::{PhysicalOp, RepresentationId};
use crate::polytope::Polytope;
use crate::star::Star2;
// use crate::starsets::Asterism;
// use crate::starsets::Constellation;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Array4;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Ix2;
use ndarray::Zip;
use ndarray::{Array, Array1};
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
        (v in array1(rows * cols))-> Array2<f64> {
            v.into_shape((rows, cols)).unwrap()
        }
}

prop_compose! {
    pub fn array3(dim_1: usize, dim_2:usize, dim_3:usize)
        (v in array1(dim_1*dim_2*dim_3))-> Array3<f64> {
            v.into_shape((dim_1, dim_2, dim_3)).unwrap()
        }
}

prop_compose! {
    pub fn array4(dim_1: usize, dim_2:usize, dim_3:usize, dim_4:usize)
        (v in array1(dim_1*dim_2*dim_3*dim_4))-> Array4<f64> {
            v.into_shape((dim_1, dim_2, dim_3, dim_4)).unwrap()
        }
}

prop_compose! {
    pub fn affine2(in_dim: usize, out_dim: usize)
        (basis in array2(out_dim, in_dim), shift in array1(out_dim)) -> Affine2 {
            Affine2::new(basis, shift)
        }
}

prop_compose! {
    pub fn bounds1(len: usize)(mut lower in array1(len), mut upper in array1(len)) -> Bounds1 {
        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| if *l > *u {mem::swap(l, u)});
        assert!(Zip::from(&lower).and(&upper).all(|l, u| l <= u));
        Bounds::new(lower.view(), upper.view())
    }
}

prop_compose! {
    pub fn bounds1_sample(bounds: Bounds1)(seed in any::<u64>()) -> Array1<f64> {
        bounds.sample_uniform(seed)
    }
}

pub fn bounds1_set(ndim: usize, half_box_width: f64) -> Bounds1 {
    assert!(half_box_width >= 0.);
    let upper_bounds = Array::ones((ndim,)) * half_box_width;
    let lower_bounds = Array::ones((ndim,)) * half_box_width * -1.;
    Bounds1::new(lower_bounds.view(), upper_bounds.view())
}

prop_compose! {
    /// Construct a sequential network
    ///
    /// # Description
    ///
    /// Constructs `n_hidden_layers` hidden layers each with between 1 and `max_layer_width` neurons.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Input dimension
    /// * `output_size` - Output dimension
    /// * `nlayers` - Number of dense/relu layer pairs
    /// * `max_layer_width` - Maximum number of dimensions in a hidden layer
    pub fn fc_dnn(input_size: usize, output_size: usize, n_hidden_layers: usize, max_layer_width: usize)
        // len(repr_sizes) == n_hidden_layers + 2
        (repr_sizes in Vec::lift1_with(1..max_layer_width+1,
                                       SizeRange::new(n_hidden_layers..=n_hidden_layers))
         .prop_map(move |mut x| {
             x.insert(0, input_size);
             x.push(output_size);
             x
         }))
        (affines in {
            let pairs = repr_sizes.iter()
                .zip(repr_sizes.iter().skip(1));
            pairs.map(|(&x, &y)| affine2(x,y)).collect::<Vec<_>>()}
        ) -> DNN {
            let mut layers: Vec<PhysicalOp> = vec![];
            for aff in affines {
                let output_dim = aff.output_dim();
                layers.push(PhysicalOp::from(Dense::new(aff.clone())));
                layers.push(PhysicalOp::from(ReLU::new(output_dim)));
            }
            DNN::from_sequential(&layers)
        }
}

// prop_compose! {
//     pub fn constellation(input_size: usize, output_size: usize, nlayers: usize, max_layer_width: usize)
//         (loc in array1(input_size), scale_diag in pos_def_array1(input_size), dnn in fc_dnn(input_size, output_size, nlayers, max_layer_width)) -> Constellation<Ix2>
//     {
//         let lbs = loc.clone() - 3.5 * scale_diag.clone();
//         let ubs = loc.clone() + 3.5 * scale_diag.clone();
//         let input_bounds = Bounds1::new(lbs.view(), ubs.view());
//         let star = Star2::new(Array2::eye(input_size), Array1::zeros(input_size));
//         Constellation::new(dnn, star, Some(input_bounds), loc, Array2::from_diag(&scale_diag).to_owned(), 4, 100, 1e-10)
//     }
// }

// prop_compose! {
//     pub fn asterism(input_size: usize, output_size: usize, nlayers: usize, max_layer_width: usize)
//         (loc in array1(input_size), scale_diag in pos_def_array1(input_size), dnn in fc_dnn(input_size, output_size, nlayers, max_layer_width)) -> Asterism<Ix2>
//     {
//         let lbs = loc.clone() - 3.5 * scale_diag.clone();
//         let ubs = loc.clone() + 3.5 * scale_diag.clone();
//         let input_bounds = Bounds1::new(lbs.view(), ubs.view());
//         let star = Star2::new(Array2::eye(input_size), Array1::zeros(input_size));
//         Asterism::new(dnn, star, loc, Array2::from_diag(&scale_diag).to_owned(), 1., Some(input_bounds), 4, 100, 1e-10)
//     }
// }

prop_compose! {
    pub fn polytope(num_dims: usize, num_constraints: usize)
        (
            mut coeffs in array2(num_constraints, num_dims),
            rhs in array1(num_constraints)
        ) -> Polytope {
            coeffs.rows_mut().into_iter().for_each(|mut row| {
                if row.iter().all(|x| *x == 0.0_f64) {
                    let mut rng = rand::thread_rng();
                    let one_idx = rng.gen_range(0..row.len());
                    row[one_idx] = 1.0;
                }
            });

            Polytope::new(coeffs, rhs)
        }
}

prop_compose! {
    pub fn polytope_including_zero(num_dims: usize, num_constraints: usize)
        (ineq in polytope(num_dims, num_constraints)) -> Polytope {
            let zero = Array1::zeros(num_dims);
            let mut polytope = Polytope::new(
                Array2::zeros((0, num_dims)),
                Array1::zeros(0));
            (0..ineq.num_constraints())
                .into_iter()
                .for_each(|eqn_idx| {
                    let eqn = ineq.get_eqn(eqn_idx);
                    if eqn.is_member(&zero.view()) {
                        polytope.add_eqn(eqn.coeffs().row(0), eqn.rhs()[[0]]);
                    } else {
                        let new_coeffs = -1. * eqn.coeffs().to_owned();
                        let new_rhs = -1. * eqn.rhs().to_owned();
                        polytope.add_eqn(new_coeffs.row(0), new_rhs[[0]]);
                    }
                    assert!(polytope.is_member(&zero.view()));
                });
            polytope
        }
}

prop_compose! {
    pub fn non_empty_polytope(num_dims: usize, num_constraints: usize)
        (
            poly in polytope_including_zero(num_dims, num_constraints)
                .prop_filter("Non-zero intercepts",
                             |i| !i.rhs().iter().any(|x| *x == 0.0_f64))
        ) -> Polytope {
            // Make a box bigger than possible inner inequalities
            //let box_bounds: Bounds1 = Bounds1::new(Array1::from_elem(num_dims, -20.).view(), Array1::from_elem(num_dims, -20.).view());
            // let polytope = Polytope::new(Array2::zeros([1, num_dims]), Array1::zeros(1));
            // assert!(!poly.is_empty(None));
            poly
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
            mut poly in polytope_including_zero(num_dims, num_constraints)
                .prop_filter("Cannot pass through origin",
                             |eq| !eq.rhs().iter().any(|x| *x == 0.0_f64))
        ) -> Polytope {
            // Invert the sign of the inequality
            poly *= -1.0;

            // Construct the inequality that is the above flipped
            // across the origin.
            let inverse_poly = Polytope::new(
                poly.coeffs().to_owned() * -1.,
                poly.rhs().to_owned() * -1.
            );
            poly.intersect(&inverse_poly);

            // Make a box bigger than possible inner inequalities
            let box_coeffs = Array2::eye(num_dims);
            let mut box_rhs = Array1::ones(num_dims);
            box_rhs *= -20.0_f64;

            let upper_box_poly = Polytope::new(box_coeffs.clone(), box_rhs.clone());
            let lower_box_poly = Polytope::new(-1. * box_coeffs, box_rhs);

            // Construct the empty polytope.
            upper_box_poly.intersect(&lower_box_poly).intersect(&poly)
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
        ) -> Star2 {
            let star = Star2::new(basis, center).with_constraints(constraints);
            assert!(!star.is_empty::<&Bounds1>(None));
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
        ) -> Star2 {
            Star2::new(basis, center).with_constraints(constraints)
        }
}

prop_compose! {
    pub fn generic_bounds1(max_len: usize)
        (dim in 1..max_len)
        (bounds in bounds1(dim)) -> Bounds1 {
            bounds
        }
}

prop_compose! {
    pub fn generic_polytope_including_zero(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (ineq in polytope_including_zero(dim, constraints)) -> Polytope {
            ineq
        }
}

prop_compose! {
    pub fn generic_empty_polytope(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (poly in empty_polytope(dim, constraints)) -> Polytope {
            poly
        }
}

prop_compose! {
    pub fn generic_non_empty_polytope(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (poly in non_empty_polytope(dim, constraints)) -> Polytope {
            poly
        }
}

prop_compose! {
    pub fn generic_empty_star(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (star in empty_star(dim, constraints)) -> Star2 {
            star
        }
}

prop_compose! {
    pub fn generic_non_empty_star(max_dims: usize, max_constraints: usize)
        (dim in 1..max_dims, constraints in 1..max_constraints)
        (star in non_empty_star(dim, constraints)) -> Star2 {
            star
        }
}

prop_compose! {
    pub fn generic_fc_dnn(max_input_size: usize, max_output_size: usize, max_nlayers: usize, max_layer_width: usize)
        (input_size in 1..max_input_size, output_size in 1..max_output_size, nlayers in 1..max_nlayers)
        (dnn in fc_dnn(input_size, output_size, nlayers, max_layer_width)) -> DNN
    {
        dnn
    }
}

// prop_compose! {
//     pub fn generic_constellation(max_input_size: usize, max_output_size: usize, max_nlayers: usize, max_layer_width: usize)
//         (input_size in 1..max_input_size, output_size in 1..max_output_size, nlayers in 1..max_nlayers)
//         (constellation in constellation(input_size, output_size, nlayers, max_layer_width)) -> Constellation<Ix2>
//         {
//             constellation
//         }
// }

// prop_compose! {
//     pub fn generic_asterism(max_input_size: usize, max_output_size: usize, max_nlayers: usize, max_layer_width: usize)
//         (input_size in 1..max_input_size, output_size in 1..max_output_size, nlayers in 1..max_nlayers)
//         (asterism in asterism(input_size, output_size, nlayers, max_layer_width)) -> Asterism<Ix2>
//         {
//             asterism
//         }
// }

proptest! {
    #[test]
    fn test_inequality_including_zero(ineq in generic_polytope_including_zero(2, 4)) {
        let zero = Array1::zeros(ineq.num_dims());
        prop_assert!(ineq.is_member(&zero.view()));
    }

    #[test]
    fn test_empty_polytope(poly in generic_empty_polytope(2, 4)) {
        prop_assert!(poly.is_empty::<&Bounds1>(None));
    }

    #[test]
    fn test_non_empty_polytope(poly in generic_non_empty_polytope(2, 4)) {
        prop_assert!(!poly.is_empty::<&Bounds1>(None));
    }

    #[test]
    fn test_non_empty_star(star in generic_non_empty_star(2, 4)) {
        prop_assert!(!star.is_empty::<&Bounds1>(None));
    }

    #[test]
    fn test_generic_fc_dnn(_dnn in generic_fc_dnn(5, 5, 5, 5)) {
        // Yes, this is the full test. The test is that we can
        // construct varying sizes of dnns.
    }
}

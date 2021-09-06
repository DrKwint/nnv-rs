#![cfg(test)]
use crate::affine::Affine2;
use crate::inequality::Inequality;
use crate::polytope::Polytope;
use crate::star::Star2;
use crate::Bounds;
use crate::Bounds1;
use crate::Layer;
use crate::DNN;
use itertools::Itertools;
use ndarray::concatenate;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Zip;
use proptest::arbitrary::functor::ArbitraryF1;
use proptest::prelude::any;
use proptest::prelude::*;
use proptest::sample::SizeRange;
use std::mem;
use std::ops::Range;

prop_compose! {
    pub fn array1(len: usize)(v in Vec::lift1_with(-10. .. 10., SizeRange::new(len..=len))) -> Array1<f64> {
        Array1::from_vec(v)
    }
}

prop_compose! {
    pub fn array2(rows: usize, cols: usize)(v in Vec::lift1_with(array1(cols), SizeRange::new(rows..=rows))) -> Array2<f64> {
        assert!(rows > 0);
        ndarray::stack(Axis(0), &v.iter().map(|x| x.view()).collect::<Vec<ArrayView1<f64>>>()).unwrap()
    }
}

prop_compose! {
    pub fn affine2(in_dim: usize, out_dim: usize)(basis in array2(out_dim, in_dim), shift in array1(out_dim)) -> Affine2<f64> {
        Affine2::new(basis, shift)
    }
}

prop_compose! {
    pub fn bounds1(len: usize)(mut lower in array1(len), mut upper in array1(len)) -> Bounds1<f64> {
        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| if *l > *u {mem::swap(l, u)});
        assert!(Zip::from(&lower).and(&upper).all(|l, u| l <= u));
        Bounds::new(lower, upper)
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
            affines.into_iter()
                .for_each(|aff| {
                    let output_dim = aff.output_dim();
                    dnn.add_layer(Layer::new_dense(aff));
                    dnn.add_layer(Layer::new_relu(output_dim))
                });
            dnn
        }
}

prop_compose! {
    pub fn polytope(num_dims: usize, num_constraints: usize)(constraint_coeffs in array2(num_dims, num_constraints), upper_bounds in array1(num_constraints))->Polytope<f64> {
        Polytope::new(constraint_coeffs, upper_bounds)
    }
}

prop_compose! {
    pub fn non_empty_star(num_dims: usize, num_constraints: usize)
        (
            basis in array2(num_dims, num_dims),
            center in array1(num_dims),
            constraints in polytope(num_dims, num_constraints)
                .prop_filter("Polytope must be feasible", |p| p.is_empty())
        ) -> Star2<f64> {
        Star2::new(basis, center).with_constraints(constraints)
    }
}

prop_compose! {
    pub fn inequality(num_dims: usize, num_constraints: usize)
        (
            coeffs in array2(num_dims, num_constraints),
            rhs in array1(num_constraints)
        )-> Inequality<f64> {
        Inequality::new(coeffs, rhs)
    }
}

prop_compose! {
    pub fn inequality_including_zero(num_dims: usize, num_constraints: usize)
        (ineq in inequality(num_dims, num_constraints)) -> Inequality<f64> {
            let zero = Array1::zeros(num_dims);
            let mut inequality = Inequality::new(
                Array2::zeros((num_dims, 0)),
                Array1::zeros(0)
            );
            (0..ineq.num_constraints())
                .into_iter()
                .for_each(|eqn_idx| {
                    let eqn = ineq.get_eqn(eqn_idx);
                    if eqn.is_member(&zero.view()) {
                        inequality.add_eqns(&eqn);
                    } else {
                        let new_coeffs = -1. * eqn.coeffs().to_owned();
                        let eqn = Inequality::new(new_coeffs, eqn.rhs().to_owned());
                        inequality.add_eqns(&eqn);
                    }
                });
            inequality
    }
}

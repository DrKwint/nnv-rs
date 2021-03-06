#![allow(clippy::module_name_repetitions)]
use crate::affine::Affine2;
use crate::rand::distributions::Distribution;
use crate::rand::SeedableRng;
use crate::NNVFloat;
use ndarray::iter::{Lanes, LanesMut};
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Ix2;
use ndarray::RemoveAxis;
use ndarray::Zip;
use ndarray::{concatenate, Array1};
use ndarray::{stack, Array, Dimension};
use ndarray::{ArrayView, ArrayViewMut, ArrayViewMut1};
use num::Float;
use num::Zero;
use ordered_float::OrderedFloat;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::Display;
use std::ops::{Mul, MulAssign};

pub type Bounds1 = Bounds<Ix2>;

#[derive(Clone, Default, Debug, PartialEq, Deserialize, Serialize)]
pub struct Bounds<D: Dimension> {
    data: Array<NNVFloat, D>,
}

impl<D: Dimension + ndarray::RemoveAxis> Bounds<D> {
    /// # Panics
    pub fn new<'a, S: Dimension + Dimension<Larger = D>>(
        lower: ArrayView<'a, NNVFloat, S>,
        upper: ArrayView<'a, NNVFloat, S>,
    ) -> Self {
        debug_assert!(
            Zip::from(&lower).and(&upper).all(|&l, &u| l <= u),
            "Input bounds are flipped!"
        );
        let data: Array<NNVFloat, D> = stack(Axis(0), &[lower, upper]).unwrap();
        Self { data }
    }

    pub fn fixed_idxs(&self) -> Array<bool, D::Smaller> {
        Zip::from(self.lower())
            .and(self.upper())
            .map_collect(|&lb, &ub| lb == ub)
    }

    pub fn fixed_vals_or_zeros(&self) -> Array<NNVFloat, D::Smaller> {
        Zip::from(self.lower())
            .and(self.upper())
            .map_collect(|&lb, &ub| if lb == ub { lb } else { NNVFloat::zero() })
    }

    pub fn fixed_vals_or_none(&self) -> Array<Option<NNVFloat>, D::Smaller> {
        Zip::from(self.lower())
            .and(self.upper())
            .map_collect(|&lb, &ub| if lb == ub { Some(lb) } else { None })
    }

    pub fn is_all_finite(&self) -> bool {
        self.data.iter().all(|&x| NNVFloat::is_finite(x))
    }

    pub fn as_tuple(&self) -> (Array<NNVFloat, D::Smaller>, Array<NNVFloat, D::Smaller>) {
        (self.lower().to_owned(), self.upper().to_owned())
    }

    pub fn lower(&self) -> ArrayView<NNVFloat, D::Smaller> {
        self.data.index_axis(Axis(0), 0)
    }

    pub fn lower_mut(&mut self) -> ArrayViewMut<NNVFloat, D::Smaller> {
        self.data.index_axis_mut(Axis(0), 0)
    }

    pub fn upper(&self) -> ArrayView<NNVFloat, D::Smaller> {
        self.data.index_axis(Axis(0), 1)
    }

    pub fn upper_mut(&mut self) -> ArrayViewMut<NNVFloat, D::Smaller> {
        self.data.index_axis_mut(Axis(0), 1)
    }

    pub fn ndim(&self) -> usize {
        self.data.shape().iter().skip(1).product()
    }

    pub fn bounds_iter(&self) -> Lanes<NNVFloat, D::Smaller> {
        self.data.lanes(Axis(0))
    }

    pub fn bounds_iter_mut(&mut self) -> LanesMut<NNVFloat, D::Smaller> {
        self.data.lanes_mut(Axis(0))
    }

    pub fn is_member(&self, x: &ArrayView<NNVFloat, D::Smaller>) -> bool {
        let eps = 1e-5;
        Zip::from(x)
            .and(self.bounds_iter())
            .all(|&x, bounds| bounds[0] - eps <= x && x <= bounds[1] + eps)
    }

    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let mut intersection = Self {
            data: self.data.clone(),
        };
        Zip::from(self.lower())
            .and(other.lower())
            .map_assign_into(intersection.lower_mut(), |&x, &y| {
                cmp::max(OrderedFloat(x), OrderedFloat(y)).0
            });
        Zip::from(self.upper())
            .and(other.upper())
            .map_assign_into(intersection.upper_mut(), |&x, &y| {
                cmp::min(OrderedFloat(x), OrderedFloat(y)).0
            });
        intersection
    }
}

impl<D: Dimension + ndarray::RemoveAxis> Bounds<D> {
    /// # Panics
    pub fn is_subset_of(&self, rhs: &Self) -> bool {
        Zip::from(self.bounds_iter())
            .and(rhs.bounds_iter())
            .all(|me, rhs| {
                let diff = me.to_owned() - rhs;
                let eps = <NNVFloat as num::NumCast>::from(1e-8).unwrap();
                (diff[[0]] >= NNVFloat::zero() || diff[[0]] <= eps)
                    && (diff[[1]] <= NNVFloat::zero() || diff[[1]] <= eps)
            })
    }
}

impl<D: Dimension + ndarray::RemoveAxis> Bounds<D> {
    pub fn sample_uniform(&self, seed: u64) -> Array<NNVFloat, D::Smaller> {
        let mut rng = StdRng::seed_from_u64(seed);
        Zip::from(self.bounds_iter())
            .map_collect(|x| Uniform::new_inclusive(x[0], x[1]).sample(&mut rng))
    }
}

impl Bounds1 {
    /// # Panics
    pub fn new_by_dim(dim_bounds: &[ArrayView1<NNVFloat>]) -> Self {
        let dims: Vec<_> = dim_bounds.iter().map(|x| x.insert_axis(Axis(1))).collect();
        Self {
            data: concatenate(Axis(1), &dims).unwrap(),
        }
    }

    pub fn default(dim: usize) -> Self {
        Self {
            data: Array2::default([2, dim]),
        }
    }

    pub fn trivial(dim: usize) -> Self {
        Self::new(
            Array::from_elem(dim, NNVFloat::neg_infinity()).view(),
            Array::from_elem(dim, NNVFloat::infinity()).view(),
        )
    }

    #[must_use]
    pub fn affine_map(&self, aff: &Affine2) -> Self {
        let lower = aff.apply(&self.lower());
        let upper = aff.apply(&self.upper());
        Self::new(lower.view(), upper.view())
    }

    pub fn split_at(&self, index: usize) -> (Self, Self) {
        let (head, tail) = self.data.view().split_at(Axis(1), index);
        (
            Self {
                data: head.to_owned(),
            },
            Self {
                data: tail.to_owned(),
            },
        )
    }

    /// # Panics
    #[must_use]
    pub fn append(mut self, other: &Self) -> Self {
        self.data.append(Axis(1), other.data.view()).unwrap();
        self
    }

    pub fn index_mut(&mut self, index: usize) -> ArrayViewMut1<NNVFloat> {
        self.data.index_axis_mut(Axis(1), index)
    }

    #[must_use]
    pub fn get_ith_bounds(&self, index: usize) -> Self {
        Self {
            data: self
                .data
                .index_axis(Axis(1), index)
                .to_owned()
                .insert_axis(Axis(0)),
        }
    }

    #[must_use]
    pub fn unfixed_dims(&self) -> Self {
        let (lower, upper): (Vec<_>, Vec<_>) = self
            .lower()
            .iter()
            .zip(self.upper().iter())
            .filter(|(&l, &u)| l != u)
            .unzip();
        Self::new(
            Array1::from_vec(lower).view(),
            Array1::from_vec(upper).view(),
        )
    }
}

/// Scale by scalar
impl<D: Dimension> Mul<NNVFloat> for Bounds<D> {
    type Output = Self;

    fn mul(self, rhs: NNVFloat) -> Self {
        Self {
            data: self.data * rhs,
        }
    }
}

/// Scale by scalar
impl<D: Dimension> MulAssign<NNVFloat> for Bounds<D> {
    fn mul_assign(&mut self, rhs: NNVFloat) {
        self.data *= rhs;
    }
}

impl<D: Dimension + RemoveAxis> Display for Bounds<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "Lower: {}\nUpper: {}", self.lower(), self.upper())
    }
}

#[cfg(test)]
mod test {
    use crate::test_util::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_bounds_sample_uniform(bounds in generic_bounds1(32)) {
            bounds.sample_uniform(0_u64);
        }
    }

    proptest! {
        #[test]
        fn test_bounds_append_shape(b_1 in generic_bounds1(32), b_2 in generic_bounds1(32)) {
            let len_b_1 = b_1.ndim();
            let len_b_2 = b_2.ndim();
            let b_3 = b_1.append(&b_2);
            prop_assert_eq!(b_3.ndim(), len_b_1 + len_b_2);
        }
    }
}

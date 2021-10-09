#![allow(clippy::module_name_repetitions)]
use crate::affine::Affine2;
use crate::rand::distributions::Distribution;
use crate::rand::SeedableRng;
use crate::NNVFloat;
use ndarray::iter::{Lanes, LanesMut};
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Ix2;
use ndarray::RemoveAxis;
use ndarray::Zip;
use ndarray::{stack, Array, Dimension};
use ndarray::{ArrayView, ArrayViewMut, ArrayViewMut1};
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use std::fmt::Display;

pub type Bounds1<T> = Bounds<T, Ix2>;

#[derive(Clone, Debug, PartialEq)]
pub struct Bounds<T: NNVFloat, D: Dimension> {
    data: Array<T, D>,
}

impl<T: NNVFloat, D: Dimension + ndarray::RemoveAxis> Bounds<T, D> {
    /// # Panics
    pub fn new<'a, S: Dimension + Dimension<Larger = D>>(
        lower: ArrayView<'a, T, S>,
        upper: ArrayView<'a, T, S>,
    ) -> Self {
        let data: Array<T, D> = stack(Axis(0), &[lower, upper]).unwrap();
        Self { data }
    }

    pub fn is_all_finite(&self) -> bool {
        self.data.iter().all(|&x| T::is_finite(x))
    }

    pub fn as_tuple(&self) -> (Array<T, D::Smaller>, Array<T, D::Smaller>) {
        (self.lower().to_owned(), self.upper().to_owned())
    }

    pub fn lower(&self) -> ArrayView<T, D::Smaller> {
        self.data.index_axis(Axis(0), 0)
    }

    pub fn lower_mut(&mut self) -> ArrayViewMut<T, D::Smaller> {
        self.data.index_axis_mut(Axis(0), 0)
    }

    pub fn upper(&self) -> ArrayView<T, D::Smaller> {
        self.data.index_axis(Axis(0), 1)
    }

    pub fn upper_mut(&mut self) -> ArrayViewMut<T, D::Smaller> {
        self.data.index_axis_mut(Axis(0), 1)
    }

    pub fn ndim(&self) -> usize {
        self.data.shape().iter().skip(1).product()
    }

    pub fn bounds_iter(&self) -> Lanes<T, D::Smaller> {
        self.data.lanes(Axis(0))
    }

    pub fn bounds_iter_mut(&mut self) -> LanesMut<T, D::Smaller> {
        self.data.lanes_mut(Axis(0))
    }

    pub fn is_member(&self, x: &ArrayView<T, D::Smaller>) -> bool {
        let eps = (1e-5).into();
        Zip::from(x)
            .and(self.bounds_iter())
            .all(|&x, bounds| bounds[0] - eps <= x && x <= bounds[1] + eps)
    }
}

impl<T: crate::NNVFloat, D: Dimension + ndarray::RemoveAxis> Bounds<T, D> {
    /// # Panics
    pub fn subset(&self, rhs: &Self) -> bool {
        Zip::from(self.bounds_iter())
            .and(rhs.bounds_iter())
            .all(|me, rhs| {
                let diff = me.to_owned() - rhs;
                let eps = <T as num::NumCast>::from(1e-8).unwrap();
                (diff[[0]] >= T::zero() || diff[[0]] <= eps)
                    && (diff[[1]] <= T::zero() || diff[[1]] <= eps)
            })
    }
}

impl<T: NNVFloat, D: Dimension + ndarray::RemoveAxis> Bounds<T, D> {
    pub fn sample_uniform(&self, seed: u64) -> Array<T, D::Smaller> {
        let mut rng = StdRng::seed_from_u64(seed);
        Zip::from(self.bounds_iter())
            .map_collect(|x| Uniform::new_inclusive(x[0], x[1]).sample(&mut rng))
    }
}

impl<T: NNVFloat> Bounds1<T> {
    pub fn default(dim: usize) -> Self {
        Self {
            data: Array2::default([2, dim]),
        }
    }

    pub fn affine_map(&self, aff: &Affine2<T>) -> Self {
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
    pub fn append(mut self, other: &Self) -> Self {
        self.data.append(Axis(1), other.data.view()).unwrap();
        self
    }

    pub fn index_mut(&mut self, index: usize) -> ArrayViewMut1<T> {
        self.data.index_axis_mut(Axis(1), index)
    }
}

impl<T: NNVFloat, D: Dimension + RemoveAxis> Display for Bounds<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "Lower: {}\nUpper: {}", self.lower(), self.upper())
    }
}

#[cfg(test)]
mod test {
    use crate::test_util::*;
    use proptest::proptest;

    proptest! {
        #[test]
        fn test_bounds_sample_uniform(bounds in generic_bounds1(32)) {
            bounds.sample_uniform(0_u64);
        }
    }
}

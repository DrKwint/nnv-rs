#![allow(clippy::module_name_repetitions)]
use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly;
use crate::dnn::DNNIndex;
use crate::dnn::DNNIterator;
use crate::dnn::DNN;
use crate::gaussian::GaussianDistribution;
use crate::star::Star;
use crate::NNVFloat;
use log::trace;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::fmt::Debug;
use truncnorm::tilting::TiltingSolution;

#[derive(Debug, Clone)]
pub enum StarNodeOp<T: NNVFloat> {
    Leaf,
    Affine(Affine2<T>),
    StepRelu(usize),
    StepReluDropout((T, usize)),
}

impl<T: NNVFloat> StarNodeOp<T> {
    /// bounds: Output bounds: Concrete bounds
    /// affs: Input bounds: Abstract bounds in terms of inputs
    ///
    /// # Panics
    pub fn apply_bounds(
        &self,
        bounds: &Bounds1<T>,
        lower_aff: &Affine2<T>,
        upper_aff: &Affine2<T>,
    ) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
        match self {
            Self::Leaf => (bounds.clone(), (lower_aff.clone(), upper_aff.clone())),
            Self::Affine(aff) => (
                aff.signed_apply(bounds),
                (
                    aff.signed_compose(lower_aff, upper_aff),
                    aff.signed_compose(upper_aff, lower_aff),
                ),
            ),
            Self::StepRelu(dim) => crate::deeppoly::deep_poly_steprelu(
                *dim,
                bounds.clone(),
                lower_aff.clone(),
                upper_aff.clone(),
            ),
            Self::StepReluDropout(_dim) => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StarNodeType<T: NNVFloat> {
    Leaf,
    Affine {
        child_idx: usize,
    },
    StepRelu {
        dim: usize,
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
    },
    StepReluDropOut {
        dim: usize,
        dropout_prob: T,
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
        trd_child_idx: Option<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct StarNode<T: NNVFloat, D: Dimension> {
    star: Star<T, D>,
    dnn_index: DNNIndex,
    star_cdf: Option<T>,
    local_bounds: Option<Bounds1<T>>,
    output_bounds: Option<(T, T)>,
    is_feasible: bool,
    gaussian_distribution: Option<GaussianDistribution<T>>,
}

impl<T: NNVFloat, D: Dimension> StarNode<T, D> {
    pub fn default(star: Star<T, D>, local_bounds: Option<Bounds1<T>>) -> Self {
        Self {
            star,
            dnn_index: DNNIndex::default(),
            star_cdf: None,
            local_bounds,
            output_bounds: None,
            is_feasible: true,
            gaussian_distribution: None,
        }
    }

    pub fn with_dnn_index(mut self, dnn_index: DNNIndex) -> Self {
        self.dnn_index = dnn_index;
        self
    }
}

impl<T: NNVFloat, D: Dimension> StarNode<T, D> {
    pub fn get_star(&self) -> &Star<T, D> {
        &self.star
    }

    pub fn get_dnn_index(&self) -> DNNIndex {
        self.dnn_index
    }

    pub fn get_feasible(&self) -> bool {
        self.is_feasible
    }

    pub fn set_feasible(&mut self, val: bool) {
        self.is_feasible = val;
    }

    pub fn set_cdf(&mut self, val: T) {
        self.star_cdf = Some(val);
    }

    pub fn reset_cdf(&mut self) {
        self.star_cdf = None;
    }

    /// # Panics
    pub fn add_cdf(&mut self, add: T) {
        if let Some(ref mut cdf) = self.star_cdf {
            *cdf += add;
            // Do this test due to cdfs being approximations
            if cdf.is_sign_negative() {
                *cdf = T::epsilon();
            }
        } else {
            // TODO
        }
    }
}

impl<T: NNVFloat> StarNode<T, Ix2> {
    /// None indicates that the distribution hasn't been calculated/constructed
    pub fn try_get_gaussian_distribution(&self) -> Option<&GaussianDistribution<T>> {
        self.gaussian_distribution.as_ref()
    }

    /// # Panics
    pub fn get_gaussian_distribution(
        &mut self,
        loc: &Array1<T>,
        scale: &Array2<T>,
        max_accept_reject_iters: usize,
        stability_eps: T,
    ) -> &mut GaussianDistribution<T> {
        if self.gaussian_distribution.is_none() {
            self.gaussian_distribution = self.star.get_input_trunc_gaussian(
                loc,
                scale,
                max_accept_reject_iters,
                stability_eps,
            );
            if self.gaussian_distribution.is_none() {
                self.gaussian_distribution = Some(GaussianDistribution::Gaussian {
                    loc: loc.clone(),
                    scale: scale.diag().to_owned(),
                });
            }
        }
        self.gaussian_distribution.as_mut().unwrap()
    }

    pub fn forward(&self, x: &Array1<T>) -> Array1<T> {
        self.star.get_representation().apply(&x.view())
    }

    pub fn get_safe_star(&self, safe_value: T) -> Self {
        let safe_star = self.star.get_safe_subset(safe_value);
        Self {
            star: safe_star,
            dnn_index: self.dnn_index,
            star_cdf: None,
            local_bounds: None,
            output_bounds: None,
            is_feasible: true,
            gaussian_distribution: None,
        }
    }

    pub fn gaussian_cdf<R: Rng>(
        &mut self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        rng: &mut R,
        stability_eps: T,
    ) -> T {
        self.star_cdf.map_or_else(
            || {
                let cdf: T = self
                    .get_gaussian_distribution(mu, sigma, max_iters, stability_eps)
                    .cdf(n, rng);
                debug_assert!(cdf.is_sign_positive());
                self.star_cdf = Some(cdf);
                cdf
            },
            |cdf| {
                debug_assert!(cdf.is_sign_positive());
                cdf
            },
        )
    }

    /// # Panics
    pub fn gaussian_sample<R: Rng>(
        &mut self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        tilting_initialization: &Option<TiltingSolution>,
        stability_eps: T,
    ) -> Vec<Array1<T>> {
        let distribution = self.get_gaussian_distribution(mu, sigma, max_iters, stability_eps);
        distribution.populate_tilting_solution(tilting_initialization.as_ref());
        distribution.sample_n(n, rng)
    }

    pub fn try_calculate_star_local_bounds(&self) -> &Option<Bounds1<T>> {
        &self.local_bounds
    }

    /// # Panics
    pub fn calculate_star_local_bounds(&mut self) -> &Bounds1<T> {
        if self.local_bounds.is_none() {
            self.local_bounds = Some(self.star.calculate_axis_aligned_bounding_box());
            debug_assert!(
                self.local_bounds
                    .clone()
                    .unwrap()
                    .bounds_iter()
                    .into_iter()
                    .all(|x| (x[[0]] <= x[[1]])),
                "Calculated bounds are flipped! {}",
                self.local_bounds.clone().unwrap()
            );
        }
        self.local_bounds.as_ref().unwrap()
    }

    pub fn get_input_bounds(&self) -> Option<&Bounds1<T>> {
        self.star.get_input_bounds()
    }

    /// # Panics
    pub fn get_output_bounds(
        &mut self,
        dnn: &DNN<T>,
        output_fn: &dyn Fn(Bounds1<T>) -> (T, T),
    ) -> (T, T) {
        if self.output_bounds.is_none() {
            trace!("get_output_bounds on star {:?}", self.star);
            let dnn_iter = DNNIterator::new(dnn, self.dnn_index);
            self.output_bounds = Some(output_fn(deep_poly(
                self.calculate_star_local_bounds(),
                dnn_iter,
            )));
        }
        self.output_bounds.unwrap()
    }
}

/// Testing getters and setters
#[cfg(test)]
impl<T: NNVFloat> StarNode<T, Ix2> {
    pub fn calculate_star_local_bounds_direct(&self) -> Option<&Bounds1<T>> {
        self.local_bounds.as_ref()
    }

    pub fn set_local_bounds_direct(&mut self, bounds: Option<Bounds1<T>>) {
        self.local_bounds = bounds
    }
}

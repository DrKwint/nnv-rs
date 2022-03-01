#![allow(clippy::module_name_repetitions)]
use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly;
use crate::dnn::dnn::DNN;
use crate::dnn::dnn_iter::DNNIndex;
use crate::dnn::dnn_iter::DNNIterator;
use crate::gaussian::GaussianDistribution;
use crate::num::Float;
use crate::polytope::Polytope;
use crate::star::Star;
use crate::NNVFloat;
use log::trace;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Dimension;
use ndarray::Ix2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use truncnorm::tilting::TiltingSolution;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StarNodeType {
    Leaf,
    Affine {
        child_idx: usize,
    },
    Conv {
        child_idx: usize,
    },
    StepRelu {
        dim: usize,
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
    },
    StepReluDropOut {
        dim: usize,
        dropout_prob: NNVFloat,
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
        trd_child_idx: Option<usize>,
    },
}

impl StarNodeType {
    pub fn get_child_ids(&self) -> Vec<usize> {
        match self {
            StarNodeType::Leaf => vec![],
            StarNodeType::Affine { child_idx } => vec![*child_idx],
            StarNodeType::Conv { child_idx } => vec![*child_idx],
            StarNodeType::StepRelu {
                dim: _,
                fst_child_idx,
                snd_child_idx,
            } => {
                let mut child_ids: Vec<usize> = vec![*fst_child_idx];
                if let Some(idx) = snd_child_idx {
                    child_ids.push(*idx);
                }
                child_ids
            }
            StarNodeType::StepReluDropOut {
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
                ..
            } => {
                let mut child_ids: Vec<usize> = vec![*fst_child_idx];
                if let Some(idx) = snd_child_idx {
                    child_ids.push(*idx);
                }
                if let Some(idx) = trd_child_idx {
                    child_ids.push(*idx);
                }
                child_ids
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StarNode<D: Dimension> {
    star: Star<D>,
    dnn_index: DNNIndex,
    star_cdf: Option<NNVFloat>,
    cdf_delta: NNVFloat,
    axis_aligned_input_bounds: Option<Bounds1>,
    output_bounds: Option<(NNVFloat, NNVFloat)>,
    gaussian_distribution: Option<GaussianDistribution>,
}

impl<D: Dimension> StarNode<D> {
    pub fn default(
        star: Star<D>,
        axis_aligned_input_bounds: Option<Bounds1>,
        initial_idx: DNNIndex,
    ) -> Self {
        Self {
            star,
            dnn_index: initial_idx,
            star_cdf: None,
            cdf_delta: 0.,
            axis_aligned_input_bounds,
            output_bounds: None,
            gaussian_distribution: None,
        }
    }

    pub fn get_star(&self) -> &Star<D> {
        &self.star
    }

    pub fn get_dnn_index(&self) -> DNNIndex {
        self.dnn_index
    }

    pub fn try_get_cdf(&self) -> Option<NNVFloat> {
        return self.star_cdf;
    }

    pub fn set_cdf(&mut self, val: NNVFloat) {
        self.star_cdf = Some(val);
    }

    pub fn reset_cdf(&mut self) {
        self.star_cdf = None;
        self.cdf_delta = 0.;
    }

    /// # Panics
    pub fn add_cdf(&mut self, add: NNVFloat) {
        self.cdf_delta += add;
    }

    pub fn try_get_output_bounds(&self) -> Option<(NNVFloat, NNVFloat)> {
        self.output_bounds
    }

    pub fn set_output_bounds(&mut self, val: (NNVFloat, NNVFloat)) {
        self.output_bounds = Some(val);
    }
}

impl StarNode<Ix2> {
    pub fn is_input_member(&self, point: &ArrayView1<NNVFloat>) -> bool {
        match self.star.input_space_polytope() {
            Some(poly) => poly.is_member(point),
            None => true,
        }
    }

    pub fn get_reduced_input_polytope(&self, bounds: &Option<Bounds1>) -> Option<Polytope> {
        self.star
            .input_space_polytope()
            .and_then(|x| x.reduce_fixed_inputs(bounds))
    }

    /// None indicates that the distribution hasn't been calculated/constructed
    pub const fn try_get_gaussian_distribution(&self) -> Option<&GaussianDistribution> {
        self.gaussian_distribution.as_ref()
    }

    pub fn set_gaussian_distribution(&mut self, val: GaussianDistribution) {
        self.gaussian_distribution = Some(val);
    }

    /// # Panics
    pub fn get_gaussian_distribution(
        &mut self,
        loc: ArrayView1<NNVFloat>,
        scale: ArrayView2<NNVFloat>,
        max_accept_reject_iters: usize,
        stability_eps: NNVFloat,
        input_bounds_opt: &Option<Bounds1>,
    ) -> &mut GaussianDistribution {
        if self.gaussian_distribution.is_none() {
            self.gaussian_distribution = self.star.get_input_trunc_gaussian(
                loc,
                scale,
                max_accept_reject_iters,
                stability_eps,
                input_bounds_opt,
            );
            if self.gaussian_distribution.is_none() {
                self.gaussian_distribution = Some(GaussianDistribution::Gaussian {
                    loc: loc.to_owned(),
                    scale: scale.diag().to_owned(),
                });
            }
        }
        self.gaussian_distribution.as_mut().unwrap()
    }

    pub fn forward(&self, x: &Array1<NNVFloat>) -> Array1<NNVFloat> {
        self.star.get_representation().apply(&x.view())
    }

    #[must_use]
    pub fn get_safe_star(&self, safe_value: NNVFloat) -> Self {
        let safe_star = self.star.get_safe_subset(safe_value);
        Self {
            star: safe_star,
            dnn_index: self.dnn_index,
            star_cdf: None,
            cdf_delta: 0.,
            axis_aligned_input_bounds: None,
            output_bounds: None,
            gaussian_distribution: None,
        }
    }

    pub fn gaussian_cdf<R: Rng>(
        &mut self,
        mu: ArrayView1<NNVFloat>,
        sigma: ArrayView2<NNVFloat>,
        n: usize,
        max_iters: usize,
        rng: &mut R,
        stability_eps: NNVFloat,
        input_bounds_opt: &Option<Bounds1>,
    ) -> NNVFloat {
        let cdf = self.star_cdf.unwrap_or_else(|| {
            let cdf: NNVFloat = self
                .get_gaussian_distribution(mu, sigma, max_iters, stability_eps, input_bounds_opt)
                .cdf(n, rng);
            debug_assert!(cdf.is_sign_positive());
            self.star_cdf = Some(cdf);
            cdf
        });
        let cdf_sum = cdf + self.cdf_delta;
        // Do this test due to cdfs being approximations
        if cdf_sum.is_sign_negative() {
            NNVFloat::epsilon()
        } else {
            cdf_sum
        }
    }

    /// # Panics
    pub fn gaussian_sample<R: Rng>(
        &mut self,
        rng: &mut R,
        mu: ArrayView1<NNVFloat>,
        sigma: ArrayView2<NNVFloat>,
        n: usize,
        max_iters: usize,
        tilting_initialization: Option<&TiltingSolution>,
        stability_eps: NNVFloat,
        input_bounds_opt: &Option<Bounds1>,
    ) -> Vec<Array1<NNVFloat>> {
        let distribution =
            self.get_gaussian_distribution(mu, sigma, max_iters, stability_eps, input_bounds_opt);
        distribution.populate_tilting_solution(tilting_initialization);
        distribution.sample_n(n, rng)
    }

    pub const fn try_get_axis_aligned_input_bounds(&self) -> &Option<Bounds1> {
        &self.axis_aligned_input_bounds
    }

    /// # Panics
    pub fn get_axis_aligned_input_bounds(&mut self, outer_bounds: &Bounds1) -> &Bounds1 {
        if self.axis_aligned_input_bounds.is_none() {
            self.axis_aligned_input_bounds = Some(
                self.star
                    .calculate_output_axis_aligned_bounding_box(&outer_bounds),
            );
        }
        self.axis_aligned_input_bounds.as_ref().unwrap()
    }

    /// # Panics
    pub fn get_output_bounds(
        &mut self,
        dnn: &DNN,
        output_fn: &dyn Fn(Bounds1) -> (NNVFloat, NNVFloat),
        outer_input_bounds: &Bounds1,
    ) -> (NNVFloat, NNVFloat) {
        if self.output_bounds.is_none() {
            trace!("get_output_bounds on star {:?}", self.star);
            let dnn_iter = DNNIterator::new(dnn, self.dnn_index);
            self.output_bounds = Some(output_fn(deep_poly(
                self.get_axis_aligned_input_bounds(&outer_input_bounds),
                dnn,
                dnn_iter,
            )));
        }
        self.output_bounds.unwrap()
    }
}

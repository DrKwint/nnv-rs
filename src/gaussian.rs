use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Ix2;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use num::One;
use rand::Rng;
use serde::{Deserialize, Serialize};
use truncnorm::distributions::MultivariateTruncatedNormal;
use truncnorm::tilting::TiltingSolution;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum GaussianDistribution {
    Gaussian {
        loc: Array1<NNVFloat>,
        scale: Array1<NNVFloat>,
    },
    TruncGaussian {
        distribution: MultivariateTruncatedNormal<Ix2>,
        inv_coeffs: Array2<NNVFloat>,
    },
}

impl GaussianDistribution {
    pub fn sample_n<R: Rng>(&mut self, n: usize, rng: &mut R) -> Vec<Array1<NNVFloat>> {
        match self {
            GaussianDistribution::TruncGaussian {
                distribution,
                inv_coeffs,
            } => {
                let sample_arr = distribution.sample_n(n, rng);
                sample_arr
                    .rows()
                    .into_iter()
                    .map(|x| (inv_coeffs.dot(&x.mapv(Into::into))))
                    .collect()
            }
            GaussianDistribution::Gaussian { ref loc, ref scale } => {
                let samples =
                    (Array2::random((n, loc.len()), StandardNormal).mapv(|x: f64| x) * scale) + loc;
                samples.rows().into_iter().map(|x| x.to_owned()).collect()
            }
        }
    }

    pub fn cdf<R: Rng>(&mut self, n: usize, rng: &mut R) -> NNVFloat {
        match self {
            GaussianDistribution::TruncGaussian { distribution, .. } => {
                let (est, _rel_err, _upper_bound) = distribution.cdf(n, rng);
                est
            }
            GaussianDistribution::Gaussian { .. } => NNVFloat::one(),
        }
    }

    pub fn try_get_tilting_solution(&self) -> Option<&TiltingSolution> {
        match self {
            GaussianDistribution::TruncGaussian { distribution, .. } => {
                distribution.try_get_tilting_solution()
            }
            GaussianDistribution::Gaussian { .. } => None,
        }
    }

    pub fn populate_tilting_solution(&mut self, initialization: Option<&TiltingSolution>) {
        if let GaussianDistribution::TruncGaussian { distribution, .. } = self {
            distribution.get_tilting_solution(initialization);
        }
    }
}

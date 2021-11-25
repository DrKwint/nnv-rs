use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Ix2;
use rand::Rng;
use truncnorm::distributions::MultivariateTruncatedNormal;
use truncnorm::tilting::TiltingSolution;

#[derive(Debug, Clone)]
pub enum GaussianDistribution<T> {
    Gaussian {
        loc: Array1<T>,
        scale: Array1<T>,
    },
    TruncGaussian {
        distribution: MultivariateTruncatedNormal<Ix2>,
        inv_coeffs: Array2<T>,
    },
}

impl<T: NNVFloat> GaussianDistribution<T> {
    pub fn sample_n<R: Rng>(&mut self, n: usize, rng: &mut R) -> Vec<Array1<T>> {
        match self {
            GaussianDistribution::TruncGaussian {
                distribution,
                inv_coeffs,
            } => {
                let sample_arr = distribution.sample_n(n, rng);
                sample_arr
                    .rows()
                    .into_iter()
                    .map(|x| (inv_coeffs.dot(&x.mapv(|x| x.into()))))
                    .collect()
            }
            GaussianDistribution::Gaussian { .. } => todo!(),
        }
    }

    pub fn cdf<R: Rng>(&mut self, n: usize, rng: &mut R) -> T {
        match self {
            GaussianDistribution::TruncGaussian { distribution, .. } => {
                let (est, _rel_err, _upper_bound) = distribution.cdf(n, rng);
                est.into()
            }
            _ => todo!(),
        }
    }

    pub fn try_get_tilting_solution(&self) -> Option<&TiltingSolution> {
        match self {
            GaussianDistribution::TruncGaussian { distribution, .. } => {
                distribution.try_get_tilting_solution()
            }
            _ => todo!(),
        }
    }

    pub fn get_tilting_solution(
        &mut self,
        initialization: Option<&TiltingSolution>,
    ) -> &TiltingSolution {
        match self {
            GaussianDistribution::TruncGaussian { distribution, .. } => {
                distribution.get_tilting_solution(initialization)
            }
            _ => todo!(),
        }
    }
}

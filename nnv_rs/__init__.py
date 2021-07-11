from .nnv_rs import *
import numpy as np
import time
from scipy.stats import norm


def bounded_sample_constellation(affines,
                                 loc,
                                 scale,
                                 safe_value,
                                 cdf_samples=100,
                                 num_samples=20,
                                 input_lower_bounds=None,
                                 input_upper_bounds=None):
    full_input_lower_bounds = np.concatenate(
        [input_lower_bounds,
         np.full(loc.shape, (-4 * scale) + loc)])
    full_input_upper_bounds = np.concatenate(
        [input_upper_bounds,
         np.full(loc.shape, (4 * scale) + loc)])
    loc = np.concatenate([input_lower_bounds, loc]).astype(np.float64)
    scale = np.diag(
        np.concatenate([np.zeros_like(input_lower_bounds),
                        scale]).astype(np.float64))
    samples, sample_logp, branch_logp = sample_constellation(
        affines, loc, scale, safe_value, cdf_samples, num_samples,
        full_input_lower_bounds, full_input_upper_bounds)
    return samples, sample_logp, branch_logp


class Constellation:
    def __init__(self, network_weights, safe_value):
        self.network_weights = network_weights
        self.safe_value = safe_value

    def set_weights(self, network_weights):
        self.network_weights = network_weights

    def bounded_sample(self, loc, scale, input_lower_bounds,
                       input_upper_bounds):
        loc = loc[0]
        if self.safe_value == np.inf:
            sample = np.random.normal(loc, scale)
            prob = 1.
            for (samp, l, s) in zip(sample, loc, scale):
                prob *= norm.pdf(samp, l, s)
            return sample, np.log(prob + 1e-12)
        try:
            sample, sample_logp, branch_logp = bounded_sample_constellation(
                self.network_weights,
                loc,
                scale,
                self.safe_value,
                input_lower_bounds=input_lower_bounds[0].astype(np.float64),
                input_upper_bounds=input_upper_bounds[0].astype(np.float64))
            prob = 1.
            for (samp, l, s) in zip(sample, loc, scale):
                prob *= norm.pdf(samp, l, s)
            normal_logp = np.log(prob)
            truncnorm_prob = np.exp(sample_logp)
            branch_prob = np.exp(branch_logp)
            if not np.all(np.isfinite(sample)):
                raise ValueError()
            return sample, (normal_logp - sample_logp) + branch_logp
        except:
            sample = np.random.normal(loc, scale)
            prob = 1.
            for (samp, l, s) in zip(sample, loc, scale):
                prob *= norm.pdf(samp, l, s)
            return sample, np.log(prob + 1e-12)

from .nnv_rs import *
import numpy as np
import time

def bounded_sample_constellation(affines,
                                 loc,
                                 scale,
                                 safe_value,
                                 cdf_samples=100,
                                 num_samples=20,
                                 input_lower_bounds=None,
                                 input_upper_bounds=None):
    start = time.perf_counter()
    loc = loc[0]
    full_input_lower_bounds = np.concatenate([input_lower_bounds, np.full(loc.shape, (-3.5 * scale) + loc)])
    full_input_upper_bounds = np.concatenate([input_upper_bounds, np.full(loc.shape, (3.5 * scale) + loc)])
    loc = np.concatenate([input_lower_bounds, loc]).astype(np.float64)
    scale = np.diag(np.concatenate([np.zeros_like(input_lower_bounds), scale]).astype(np.float64))
    out = sample_constellation(affines, loc, scale, safe_value, cdf_samples,
                         num_samples, full_input_lower_bounds, full_input_upper_bounds)
    end = time.perf_counter()
    print("Time (sec): ", end - start)
    return out


class Constellation:
    def __init__(self, network_weights, safe_value):
        self.network_weights = network_weights
        self.safe_value = safe_value

    def set_weights(self, network_weights):
        self.network_weights = network_weights

    def bounded_sample(self, loc, scale, input_lower_bounds,
                       input_upper_bounds):
        if self.safe_value == np.inf:
            return np.random.normal(loc, scale)[0]
        return bounded_sample_constellation(self.network_weights,
                                     loc,
                                     scale,
                                     self.safe_value,
                                     input_lower_bounds=input_lower_bounds[0].astype(np.float64),
                                     input_upper_bounds=input_upper_bounds[0].astype(np.float64))

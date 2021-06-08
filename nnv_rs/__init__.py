from .nnv_rs import *


def bounded_sample_constellation(affines,
                                 loc,
                                 scale,
                                 safe_value,
                                 cdf_samples=1000,
                                 num_samples=10,
                                 input_lower_bounds=None,
                                 input_upper_bounds=None):
    sample_constellation(affines, loc, scale, safe_value, cdf_samples,
                         num_samples, input_lower_bounds, input_upper_bounds)


class Constellation:
    def __init__(self, network_weights):
        self.network_weights = network_weights

    def set_weights(self, network_weights):
        self.network_weights = network_weights

    def bounded_sample(self, loc, scale, safe_value, input_lower_bounds,
                       input_upper_bounds):
        bounded_sample_constellation(self.network_weights,
                                     loc,
                                     scale,
                                     safe_value,
                                     input_lower_bounds,
                                     input_lower_bounds=input_lower_bounds,
                                     input_upper_bounds=input_upper_bounds)

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

from .nnv_rs import PyConstellation, PyDNN
import numpy as np
import tree
from scipy.stats import norm


class DNN:
    def __init__(self, network):
        self.dnn = PyDNN()
        type_str = str(type(network))
        if 'tensorflow' in type_str:
            import tensorflow as tf
            assert isinstance(network, tf.Module)
            self.build_from_tensorflow_module(network)
        else:
            raise NotImplementedError()

    def build_from_tensorflow_module(self, network):
        from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

        submodules = network.layers
        submodules = tree.flatten(submodules)
        while any([x.submodules for x in submodules]):
            submodules = [
                x.submodules if x.submodules else x for x in submodules
            ]
            submodules = tree.flatten(submodules)
        for layer in submodules:
            # Call the appropriate rust-code defined function
            if isinstance(layer, InputLayer):
                pass  # we don't need to do anything in this case
            elif isinstance(layer, Dense):
                weights = layer.get_weights()
                self.dnn.add_dense(weights[0].T, weights[1])
            elif isinstance(layer, Conv2D):
                weights = layer.get_weights()
                self.dnn.add_conv(weights[0], weights[1])
            elif isinstance(layer, MaxPooling2D):
                pool_size = layer.pool_size[0]
                assert layer.pool_size[0] == layer.pool_size[1]
                self.dnn.add_maxpool(pool_size)
            elif isinstance(layer, Flatten):
                self.dnn.add_flatten()
            else:
                raise NotImplementedError()

    def __str__(self):
        return str(self.dnn)


class Constellation:
    def __init__(self, network, input_bounds=None, safe_value=np.inf):
        print(type(network.dnn))
        self.constellation = PyConstellation(network.dnn, input_bounds)
        self.safe_value = safe_value

    def _weights(self, network_weights):
        self.network_weights = network_weights

    def importance_sample(self, loc, scale):
        pass

    def bounded_sample(self, loc, scale, input_lower_bounds,
                       input_upper_bounds):
        if self.safe_value == np.inf:
            sample = np.random.normal(loc, scale)
            prob = 1.
            for (samp, l, s) in zip(sample, loc, scale):
                prob *= norm.pdf(samp, l, s)
            return sample, np.log(prob + 1e-12)
        sample, sample_logp, branch_logp = self.constellation.bounded_sample_multivariate_gaussian(
            loc,
            scale,
            self.safe_value,
            cdf_samples=100,
            num_samples=20,
            max_iters=10)
        prob = 1.
        for (samp, l, s) in zip(sample, loc, scale):
            prob *= norm.pdf(samp, l, s)
        normal_logp = np.log(prob)
        truncnorm_prob = np.exp(sample_logp)
        branch_prob = np.exp(branch_logp)
        if not np.all(np.isfinite(sample)):
            raise ValueError()
        return sample, (normal_logp - sample_logp) + branch_logp
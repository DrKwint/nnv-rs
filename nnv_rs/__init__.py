from nnv_rs.nnv_rs import PyConstellation, PyDNN
import numpy as np
import tree
from scipy.stats import norm

SAMPLE_STD_DEVS = 3.5


class DNN:
    def __init__(self, network):
        self.dnn = PyDNN()
        type_ = type(network)
        if 'tensorflow' in str(type_):  # TF2
            import tensorflow as tf
            assert isinstance(network, tf.Module)
            self.build_from_tensorflow_module(network)
        elif type_ == list:  # TF1
            # Assume that inputs are tuples (weights, bias) and that each laye
            assert (type(network[0]) == tuple)
            assert (type(network[0][0]) == np.ndarray)
            self.build_from_tensorflow_params(network)
        else:
            raise NotImplementedError(str(type_))

    def input_shape(self):
        return self.dnn.input_shape()

    def deeppoly_bounds(self, lower, upper):
        return self.dnn.deeppoly_output_bounds(lower, upper)

    def build_from_tensorflow_params(self, affine_list):
        nlayers = len(affine_list)
        for i, aff in enumerate(affine_list):
            # Add dense
            self.dnn.add_dense(aff[0].T, aff[1])
            # Add relu
            if i != nlayers - 1:
                self.dnn.add_relu(aff[1].shape[0])

    def build_from_tensorflow_module(self, network):
        import tensorflow as tf
        from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, ReLU

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
                if layer.activation == tf.nn.relu:
                    self.dnn.add_relu(len(weights[1]))
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
                print('Unknown layer', layer)
                raise NotImplementedError()

    def __str__(self):
        return str(self.dnn)


class Constellation:
    def __init__(self, dnn, loc, scale, input_bounds=None, safe_value=np.inf):
        self.loc = np.squeeze(loc).astype(np.float64)
        self.scale = np.squeeze(scale).astype(np.float64)
        if dnn is None:
            self.constellation = None
        else:
            self.constellation = PyConstellation(dnn.dnn, input_bounds,
                                                 self.loc, self.scale)
        self.safe_value = safe_value

    def set_dnn(self, dnn, loc, scale):
        self.loc = np.squeeze(loc).astype(np.float64)
        self.scale = np.squeeze(scale).astype(np.float64)
        if self.constellation is None:
            bounds = None
        else:
            bounds = self.constellation.get_input_bounds()
        self.constellation = PyConstellation(dnn.dnn, bounds, loc, scale)

    def set_input_bounds(self, fixed_part, loc, scale):
        unfixed_part = (loc - SAMPLE_STD_DEVS * scale,
                        loc + SAMPLE_STD_DEVS * scale)
        self.constellation.set_input_bounds(fixed_part, unfixed_part)

    def importance_sample(self):
        pass

    def bounded_sample_with_input_bounds(self, fixed_part):
        fixed_part = np.squeeze(fixed_part).astype(np.float64)
        self.set_input_bounds(fixed_part, self.loc, self.scale)
        return self.bounded_sample(self.loc, self.scale)

    def bounded_sample(self):
        if self.safe_value == np.inf:
            sample = np.random.normal(self.loc[-len(self.scale):], self.scale)
            prob = 1.
            for (samp, l, s) in zip(sample, self.loc, self.scale):
                prob *= norm.pdf(samp, l, s)
            return sample, np.log(prob + 1e-12)
        sample, sample_logp, branch_logp = self.constellation.bounded_sample_multivariate_gaussian(
            self.safe_value, cdf_samples=100, num_samples=20, max_iters=10)
        prob = 1.
        for (samp, l, s) in zip(sample, self.loc, self.scale):
            prob *= norm.pdf(samp, l, s)
        normal_logp = np.log(prob)
        truncnorm_prob = np.exp(sample_logp)
        branch_prob = np.exp(branch_logp)
        if not np.all(np.isfinite(sample)):
            raise ValueError()
        return sample, (normal_logp - sample_logp) + branch_logp

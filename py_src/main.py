from numpy.core.defchararray import lower
import tensorflow as tf
import nnv_rs
import numpy as np
import time

print("HELLO WORLD")
l1 = tf.keras.layers.Dense(8)
l2 = tf.keras.layers.Dense(1)
net = tf.keras.models.Sequential([l1, l2])

net(tf.ones([1, 16]))
weights = [tuple(l.get_weights()) for l in net.layers]
print([(a.shape, b.shape) for (a, b) in weights])
lower_bounds = np.random.standard_normal(16)
upper_bounds = lower_bounds + np.concatenate(
    [np.zeros(14), np.abs(np.random.standard_normal(2))])
before = time.perf_counter()
samples = nnv_rs.sample_constellation(weights,
                                      loc=np.ones(16),
                                      scale=np.eye(16),
                                      safe_value=0.67,
                                      cdf_samples=100,
                                      num_samples=10,
                                      input_lower_bounds=lower_bounds,
                                      input_upper_bounds=upper_bounds)
after = time.perf_counter()
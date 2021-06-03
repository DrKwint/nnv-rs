import tensorflow as tf
import nnv_rs
import numpy as np
import time

l1 = tf.keras.layers.Dense(16)
l2 = tf.keras.layers.Dense(1)
net = tf.keras.models.Sequential([l1, l2])

net(tf.ones([1, 16]))
weights = [tuple(l.get_weights()) for l in net.layers]
print([(a.shape, b.shape) for (a, b) in weights])
before = time.perf_counter()
samples = nnv_rs.sample_constellation(weights,
                                      loc=np.ones(16),
                                      scale=np.eye(16),
                                      cdf_samples=100,
                                      num_samples=10)
after = time.perf_counter()
print(type(samples))
print(len(samples))
print(samples)
print(after - before)

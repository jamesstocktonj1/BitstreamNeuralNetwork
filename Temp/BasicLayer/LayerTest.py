import numpy as np
import tensorflow as tf
from Model import BitModel
from Layer import BitLayer


x = tf.ones((2, 4), dtype=np.int8)
w = tf.ones((2, 3)) * 0.2

layer = BitLayer(3, 4)
y = layer(x)

print(x)
print(y)
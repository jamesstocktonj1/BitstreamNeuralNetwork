import tensorflow as tf
import numpy as np
from tensorflow.keras import layers




class BitLayer(layers.Layer):

    def __init__(self, num_outputs, bit_size):
        super(BitLayer, self).__init__()

        self.num_outputs = num_outputs
        self.bit_size = bit_size

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.num_outputs, self.bit_size), dtype=np.int8, trainable=True)

    def call(self, x):

        # tensor -> np.array
        x = np.array(x)

        # bitstream mulitplication
        x = x & np.array(self.kernel)

        # sum columns
        x = x.sum(axis=1)

        # normalise values
        x = (x > 0) * 1

        return x


class BitIntegrator(layers.Layer):
    def __init__(self):
        super(BitIntegrator, self).__init__()

    def call(self, x):

        # tensor -> np.array
        x = np.array(x)
        y = np.zeros(shape=(x.shape[0]))

        for i in range(x.shape[0]):
            y[i] = x[i].sum() / x[i].size

        # return vectorised array
        return y
        
class BitCreator(layers.Layer):
    def __init__(self, bit_size):
        super(BitCreator, self).__init__()

        self.bit_size = bit_size

    def call(self, x):
        x = tf.zeros(x.shape + (self.bit_size))
        return x
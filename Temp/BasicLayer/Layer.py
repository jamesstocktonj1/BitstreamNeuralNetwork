import tensorflow as tf
import numpy as np
from tensorflow.keras import layers




class BitLayer(layers.Layer):

    def __init__(self, num_outputs, bit_size):
        super(BitLayer, self).__init__()

        self.num_outputs = num_outputs
        self.bit_size = bit_size

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-2], self.num_outputs), trainable=True, initializer=tf.keras.initializers.Zeros())

    def create_weights(self):
        w = np.zeros(self.kernel.shape + (self.bit_size), dtype=np.int8)

        for x in range(self.kernel.shape[0]):
            for y in range(self.kernel.shape[1]):
                if type(tf.get_static_value(self.kernel[x,y])) != type(None):
                    w[x,y] = np.random.binomial(1, self.kernel[x,y], self.bit_size)

        return w

    def call(self, x):
        
        x = tf.cast(x, tf.int8)
        x = tf.get_static_value(x)

        #x = x.numpy()
        #x = np.array(x)
        y = np.zeros(shape=(self.num_outputs, self.bit_size))

        # create weights
        w = self.create_weights()

        # itterate through output
        for n in range(self.kernel.shape[-1]):
            temp = np.array(x & w[:,n])
            temp = temp.sum(axis=0)
            temp = (temp > 0) * 1
            y[n] = temp

        return tf.constant(y)


class BitIntegrator(layers.Layer):
    def __init__(self):
        super(BitIntegrator, self).__init__()

    def call(self, x):

        # tensor -> np.array
        x = tf.get_static_value(x)
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

        y = np.zeros(x.shape + (self.bit_size))
        x = tf.get_static_value(x)

        if type(x) == type(None):
            return tf.constant(y)

        for i in range(x.shape[0]):
            y[i] = np.random.binomial(1, x[i], self.bit_size)

        return tf.constant(y)
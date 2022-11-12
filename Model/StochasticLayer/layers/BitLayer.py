import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import Layer
from layers.BitInput import bitstream_generator_exact





class BitLayer(Layer):
    def __init__(self, num_outputs, bit_size):
        super(BitLayer, self).__init__()

        self.num_outputs = num_outputs
        self.bit_size = bit_size

    def build(self, input_shape):
        # input_shape[-1] is bit_size
        # input_shape[-2] is number of input streams
        self.kernel = self.add_weight(shape=(self.num_outputs, input_shape[-2]), trainable=True, initializer=tf.keras.initializers.Zeros())

    def generate_weights(self):

        # converts array of floats into bitstream (recursive unpacking)
        def generate_bin(inputs):
            if inputs.get_shape().ndims != 0:
                return tf.map_fn(generate_bin, inputs)
            else:
                return bitstream_generator_exact(inputs, self.bit_size)

        return generate_bin(self.kernel)

    def call(self, inputs):

        w = self.generate_weights()

        def compute_mult(weight, inputs):
            # cast to and-able form
            weight = tf.cast(weight, dtype=tf.int16)
            inputs = tf.cast(inputs, dtype=tf.int16)

            # "multiply" inputs and weight
            temp = weight & inputs

            # "sum" values together
            temp = tf.reduce_sum(temp, 0)

            # normalise values
            temp = tf.cond(temp > 0, lambda: 1, lambda: 0)

            return temp

            # return tf.constant(temp, dtype=tf.float32)


        def compute_output(weight, inputs):
            # itterate through output size
            return tf.map_fn(lambda w: compute_mult(w, inputs), weight, dtype=tf.float32)

        output = None
            
        # if batch data then unpack
        if inputs.get_shape().ndims == 3:
            output = tf.map_fn(lambda x: compute_output(w, x), inputs)
        else:
            output = compute_output(w, inputs)

        # compute bitstream value for loss
        y_output = tf.math.reduce_sum(output, axis=tf.rank(output) - 1) / self.bit_size
        self.add_loss(y_output)

        return output

def bitlayer_test():

    N = 4
    M = 5
    layer = BitLayer(M, N)
    
    x = tf.ones((M-1, N))
    y = layer(x)
    print(x)
    print(y)
    
    print(layer.generate_weights())
    
    layer.kernel = tf.Variable([
        [0.5, 0.25, 0.3, 0.7],
        [0.5, 0.25, 0.3, 0.12],
        [0.5, 0.75, 0.23, 0.12],
        [0.5, 0.25, 0.3, 0.12],
        [0.45, 0.15, 0.3, 0.12]
    ])

    x = tf.ones((M-1, N))
    y = layer(x)
    print(x)
    print(y)

if __name__ == "__main__":
    bitlayer_test()
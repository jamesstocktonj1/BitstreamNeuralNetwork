import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import Layer


class BitInput(Layer):
    def __init__(self, bit_size):
        super(BitInput, self).__init__()

        self.bit_size = bit_size

    def call(self, input):
        
        # converts array of floats into bitstream (recursive unpacking)
        def generate_bin(inputs):
            if inputs.get_shape().ndims != 0:
                return tf.map_fn(generate_bin, inputs)
            else:
                return np.random.binomial(1, inputs, self.bit_size)

        return generate_bin(input)


def bitinput_test():

    N = 16
    layer = BitInput(N)

    # 1D to 2D
    x1 = tf.constant(
        [0.25, 0.35, 0.5]
    )
    y1 = layer(x1)
    print("1D to 2D Convert")
    print(x1)
    print(y1)

    # 2D to 3D
    x2 = tf.constant([
        [0.25, 0.35, 0.5],
        [0.15, 0.75, 0.3]
    ])
    y2 = layer(x2)
    print("2D to 3D Convert")
    print(x2)
    print(y2)

if __name__ == "__main__":
    bitinput_test()
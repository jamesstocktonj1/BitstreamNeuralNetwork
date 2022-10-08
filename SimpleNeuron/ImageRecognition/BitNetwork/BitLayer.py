import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Bit Neuron Perceptron
class BitNeurons(layers.Layer):

    def __init__(self, num_outputs):
        super(BitNeurons, self).__init__()

        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from BitLayer import BitNeurons


# create model
class BitModel(Model):

    def __init__(self):
        super(BitModel, self).__init__()

        self.input_layer = BitNeurons(128)
        self.dense1 = layers.Dense(128, activation=tf.nn.relu)

        self.softmax = layers.Dense(10, activation=tf.nn.softmax)

    def call(self, x, is_training=False):
        x = self.input_layer(x)

        x = self.dense1(x)

        if not is_training:
            x = self.softmax(x)

        return x
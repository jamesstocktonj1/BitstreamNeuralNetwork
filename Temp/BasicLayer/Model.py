import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from Layer import BitIntegrator, BitLayer



class BitModel(Model):

    def __init__(self):
        super(BitModel, self).__init__()

        self.dense1 = BitLayer(2, 4)
        self.integrator = BitIntegrator()

    def call(self, x):
        x = self.dense1(x)
        x = self.integrator(x)
        return x
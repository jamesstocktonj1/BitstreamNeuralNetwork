import numpy as np




class Neuron:
    def __init__(self, input_shape):
        self.w = np.zeros((input_shape))

    def set_weights(self, w):
        self.w = w

    def call(self, x):
        mult = x * self.w
        return 1 - np.product(mult)
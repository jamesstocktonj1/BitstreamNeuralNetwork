import numpy as np




class Neuron:
    def __init__(self, input_shape):
        self.weights = np.zeros((input_shape))

    def set_weights(self, w):
        self.weights = w

    def call(self, x):
        mult = x * self.weights
        return 1 - np.product(1 - mult)

    def decision(self, x):
        return (0.5 - (x * self.weights[0])) / (self.weights[1] - (x * self.weights[0] * self.weights[1]))
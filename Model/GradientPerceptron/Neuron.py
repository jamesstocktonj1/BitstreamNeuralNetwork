import numpy as np




class Neuron:
    def __init__(self, input_shape):
        self.weights = np.zeros((input_shape))

    def set_weights(self, w):
        self.weights = w

    def call(self, x):
        mult = x * self.weights
        return 1 - np.product(1 - mult)

    def diff(self, y_hat, x):
        w = self.weights

        dw0 = -2 * x[0] * (1 - (x[1]*w[1])) * (y_hat - 1 + ((1 - (w[0]*x[0])) * (1 - (w[1]*x[1]))))
        dw1 = -2 * x[1] * (1 - (x[0]*w[0])) * (y_hat - 1 + ((1 - (w[0]*x[0])) * (1 - (w[1]*x[1]))))

        return np.array([dw0[0], dw1[0]])

    def decision(self, x):
        return (0.5 - (x * self.weights[0])) / (self.weights[1] - (x * self.weights[0] * self.weights[1]))
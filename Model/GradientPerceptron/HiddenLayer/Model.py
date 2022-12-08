import numpy as np





class HiddenPerceptron:
    def __init__(self):
        self.w = np.zeros((6, ))

    def call(self, x):
        z_1, z_2 = self.calc_layer(x)
        y =   1 - ((1 - (z_1  * self.w[4])) * (1 - (z_2  * self.w[5])))

        return y

    def calc_layer(self, x):
        z_1 = 1 - ((1 - (x[0] * self.w[0])) * (1 - (x[1] * self.w[1])))
        z_2 = 1 - ((1 - (x[0] * self.w[2])) * (1 - (x[1] * self.w[3])))
        return z_1, z_2

    def back_prop(self, x, y):

        z_1, z_2 = self.calc_layer(x)

        dloss = y - 1 + ((1 - (z_1 * self.w[4])) * (1 - (z_2 * self.w[5])))

        dz1 = (1 - (z_2 * self.w[5])) * dloss
        dz2 = (1 - (z_1 * self.w[4])) * dloss

        dw1 = 2 * x[0] * z_1 * dz1
        dw2 = 2 * x[1] * z_1 * dz1
        dw3 = 2 * x[0] * z_2 * dz2
        dw4 = 2 * x[1] * z_2 * dz2

        dw5 = -2 * z_1 * dz1
        dw6 = -2 * z_2 * dz2

        return np.array([dw1, dw2, dw3, dw4, dw5, dw6]).reshape(6)

    def set_weights(self, w):
        self.w = w


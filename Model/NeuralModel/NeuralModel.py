import numpy as np

from NeuralLayer import NeuralLayer




class SimpleModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(2, 2)
        self.layer2 = NeuralLayer(2, 1)

    def init_weights(self):
        
        # self.layer1.weights = np.array([
        #     np.random.randint(20, 60, size=2) / 100,
        #     np.random.randint(20, 80, size=2) / 100,
        # ])

        # self.layer2.weights = np.array([
        #     np.random.randint(50, 90, size=2) / 100,
        # ])

        self.layer1.weights = np.interp(np.random.randn(2, 2), (-1, 1), (0, 1))
        self.layer2.weights = np.interp(np.random.randn(1, 2), (-1, 1), (0, 1))


    def grad(self, x, y):
        z = self.layer1.call(x)
        grad1 = self.layer1.grad(x)
        grad2 = self.layer2.loss_grad(z, y)

        self.layer1.weights -= self.training_rate * grad1 * grad2
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)

        self.layer2.weights -= self.training_rate * grad2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)

        return grad1 * grad2.transpose(), grad2
        
        
    def call(self, x):
        z = self.layer1.call(x)
        return self.layer2.call(z)


class DoubleModel:
    def __init__(self, R):
        self.training_rate = R

        np.random.seed(np.random.MT19937(np.random.SeedSequence(123456789)))

        self.layer1 = NeuralLayer(2, 2)
        self.layer2 = NeuralLayer(2, 2)

    def init_weights(self):
        self.layer1.weights = np.interp(np.random.randn(2, 2), (-1, 1), (0, 1))
        self.layer2.weights = np.interp(np.random.randn(2, 2), (-1, 1), (0, 1))
        
        # self.layer1.weights = np.array([
        #     np.random.randint(20, 60, size=2) / 100,
        #     np.random.randint(20, 80, size=2) / 100,
        # ])

        # self.layer2.weights = np.array([
        #     np.random.randint(50, 90, size=2) / 100,
        #     np.random.randint(50, 90, size=2) / 100,
        # ])


    def grad(self, x, y):
        z = self.layer1.call(x)
        grad1 = self.layer1.grad(x)
        grad2 = self.layer2.loss_grad(z, y)

        self.layer1.weights -= self.training_rate * grad1 * grad2
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)

        self.layer2.weights -= self.training_rate * grad2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)

        return grad1 * grad2, grad2
        
        
    def call(self, x):
        z = self.layer1.call(x)
        return self.layer2.call(z)


class BigModel:

    def __init__(self, R):
        self.training_rate = R
        
        self.layer1 = NeuralLayer(2, 4)
        self.layer2 = NeuralLayer(4, 2)
        self.layer3 = NeuralLayer(2, 1)

    def init_weights(self):
        # self.layer1.weights = np.interp(np.random.randn(4, 2), (-1, 1), (0, 1))
        # self.layer2.weights = np.interp(np.random.randn(2, 4), (-1, 1), (0, 1))
        # self.layer3.weights = np.interp(np.random.randn(1, 2), (-1, 1), (0, 1))

        self.layer1.weights = 0.5 + 0.1 * np.random.randn(4, 2)
        self.layer2.weights = 0.5 + 0.1 * np.random.randn(2, 4)
        self.layer3.weights = 0.5 + 0.1 * np.random.randn(1, 2)

        # print(self.layer1.weights)

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)

        grad1 = self.layer1.grad(x) # dx / wi
        grad2 = self.layer2.grad(z1) # dz1 / wi
        grad3 = self.layer3.loss_grad(z2, y) # dL / dwi

        

        # dL / dw2i = (dL / dz2) * (dz1 / dw2i)
        # dL / dw1i = (dL / dw2i) * (dw2i / dw1i)     (dx / dw1i) * () * (dL / d)



        # dL / dw1i = (dL / dz2) * (dz2 / dw2i) * (dw2i / dw1i)
        # dw2i / dw1i = (dw2i / dwz1) * (dwz1 / dw1i)
        # dL / dw1i = (dL / dz2) * (dz2 / dw2i) * (dw2i / dwz1) * (dwz1 / dw1i)

        # print(grad1)
        # print(grad2)
        # print(grad3)

        # 5 / 0

        dw1 = grad3 * grad2.transpose() * grad1
        dw2 = grad3.transpose() * grad2
        dw3 = grad3

        print(dw1)
        print(dw2)
        print(dw3)

        5 / 0

        self.layer1.weights -= self.training_rate * dw1
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)

        self.layer2.weights -= self.training_rate * dw2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)

        self.layer3.weights -= self.training_rate * dw3
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)

        return dw1, dw2, dw3

    def call(self, x):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        return self.layer3.call(z2)
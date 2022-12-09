import numpy as np

from NeuralLayer import NeuralLayer




class SimpleModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(2, 2)
        self.layer2 = NeuralLayer(2, 1)

    def init_weights(self):
        
        self.layer1.weights = np.array([
            np.random.randint(40, 60, size=2) / 100,
            np.random.randint(40, 60, size=2) / 100,
        ])

        self.layer2.weights = np.array([
            np.random.randint(40, 60, size=2) / 100,
        ])


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

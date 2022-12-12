import numpy as np


from NeuralLayer import NeuralLayer




class SimpleModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(2, 4)
        self.layer2 = NeuralLayer(4, 1)

        self.layer1.init_weights(0.1)
        self.layer2.init_weights(0.1)

    def grad(self, x, y):
        z = self.layer1.call(x)
        y_hat = self.layer2.call(z)
        
        dLoss = self.layer2.grad_loss(z, y)
        dLayer2 = self.layer2.grad_layer(z, y_hat)
        dWeight2 = self.layer2.grad_weight(z)

        # print("Loss ", dLoss)
        # print("Layer ", dLayer2)
        # print("Weight: ", dWeight2)

        dWeight1 = self.layer1.grad_weight(x)

        # print("Weight: ", dWeight1)

        dW2 = dLoss * dWeight2
        dW1 = np.dot(dLoss * dLayer2, dWeight1)

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)

        return dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        return self.layer2.call(z)
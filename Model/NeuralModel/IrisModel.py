from queue import PriorityQueue
import numpy as np
from NeuralLayer import NeuralLayer, NeuralLayerBias


class IrisModel:

    def __init__(self, R, L, U):
        self.training_rate = R
        self.regularisation = L

        self.crossentropy = True

        self.layer1 = NeuralLayerBias(4, 5)
        self.layer2 = NeuralLayerBias(5, 5)
        self.layer3 = NeuralLayerBias(5, 3)

        self.layer1.init_weights(U)
        self.layer2.init_weights(U)
        self.layer3.init_weights(U)

        self.layer1.relu = True
        self.layer2.relu = True

        self.layer3.crossentropy = self.crossentropy
        self.layer3.no_activation = True

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        y_hat = self.layer3.call(z2)

        dLoss = self.layer3.grad_loss(y_hat, y)
        dLayer3 = self.layer3.grad_layer(z2, y_hat)
        dWeight3 = self.layer3.grad_weight(z2)
        dBias3 = self.layer3.grad_bias(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)
        dBias2 = self.layer2.grad_bias(z1)

        dWeight1 = self.layer1.grad_weight(x)
        dBias1 = self.layer1.grad_bias(x)

        dW3 = dLoss.T * dWeight3
        dW2 = np.dot(dLoss, dLayer3).T * dWeight2
        dW1 = np.dot(np.dot(dLoss, dLayer3), dLayer2).T * dWeight1

        dB3 = dLoss.T * dBias3
        dB2 = np.dot(dLoss, dLayer3).T * dBias2
        dB1 = np.dot(np.dot(dLoss, dLayer3), dLayer2).T * dBias1

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights -= self.regularisation * self.layer1.weights
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights -= self.regularisation * self.layer2.weights
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights -= self.regularisation * self.layer3.weights
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)

        self.layer1.bias -= self.training_rate * dB1
        self.layer1.bias -= self.regularisation * self.layer1.bias
        self.layer1.bias = np.clip(self.layer1.bias, 0.0, 1.0)
        self.layer2.bias -= self.training_rate * dB2
        self.layer2.bias -= self.regularisation * self.layer2.bias
        self.layer2.bias = np.clip(self.layer2.bias, 0.0, 1.0)
        self.layer3.bias -= self.training_rate * dB3
        self.layer3.bias -= self.regularisation * self.layer3.bias
        self.layer3.bias = np.clip(self.layer3.bias, 0.0, 1.0)

        return dW3, dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)
        
        if not self.crossentropy:
            return (y - y_hat) ** 2
        else:
            return -1 * (y * np.log(y_hat)).sum()

    def call(self, x):
        print("\n\nModel Call")
        z = self.layer1.call(x)
        print("Layer 1: {}".format(z))
        z = self.layer2.call(z)
        print("Layer 2: {}".format(z))
        z = self.layer3.call(z)
        print("Layer 3: {}".format(z))
        return z


class IrisModel2:

    def __init__(self, R, L, U):
        self.training_rate = R
        self.regularisation = L

        self.crossentropy = False

        self.layer1 = NeuralLayer(5, 10)
        self.layer2 = NeuralLayer(10, 10)
        self.layer3 = NeuralLayer(10, 10)
        self.layer4 = NeuralLayer(10, 5)
        self.layer5 = NeuralLayer(5, 3)

        self.layer1.init_xavier(U)
        self.layer2.init_xavier(U)
        self.layer3.init_xavier(U)
        self.layer4.init_xavier(U)
        self.layer5.init_xavier(U)

        self.layer1.no_activation = True
        self.layer2.no_activation = True
        self.layer3.relu = True
        self.layer4.relu = True
        self.layer5.softmax = True

        self.layer5.crossentropy = self.crossentropy

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        z3 = self.layer3.call(z2)
        z4 = self.layer4.call(z3)
        y_hat = self.layer5.call(z4)

        dLoss = self.layer5.grad_loss(y_hat, y)
        dLayer5 = self.layer5.grad_layer(z4, y_hat)
        dWeight5 = self.layer5.grad_weight(z4)

        dLayer4 = self.layer4.grad_layer(z3, z4)
        dWeight4 = self.layer4.grad_weight(z3)

        dLayer3 = self.layer3.grad_layer(z2, z3)
        dWeight3 = self.layer3.grad_weight(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)

        dWeight1 = self.layer1.grad_weight(x)

        dW5 = dLoss.T * dWeight5
        dW4 = (dLoss @ dLayer5).T * dWeight4
        dW3 = ((dLoss @ dLayer5) @ dLayer4).T * dWeight3
        dW2 = (((dLoss @ dLayer5) @ dLayer4) @ dLayer3).T * dWeight2
        dW1 = ((((dLoss @ dLayer5) @ dLayer4) @ dLayer3) @ dLayer2).T * dWeight1

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights -= self.regularisation * self.layer1.weights
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights -= self.regularisation * self.layer2.weights
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights -= self.regularisation * self.layer3.weights
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)
        self.layer4.weights -= self.training_rate * dW4
        self.layer4.weights -= self.regularisation * self.layer4.weights
        self.layer4.weights = np.clip(self.layer4.weights, 0.0, 1.0)
        self.layer5.weights -= self.training_rate * dW5
        self.layer5.weights -= self.regularisation * self.layer5.weights
        self.layer5.weights = np.clip(self.layer5.weights, 0.0, 1.0)

        return dW5, dW4, dW3, dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)
        
        if not self.crossentropy:
            return (y - y_hat) ** 2
        else:
            return -1 * (y * np.log(y_hat)).sum()

    def call(self, x):
        z = self.layer1.call(x)
        z = self.layer2.call(z)
        z = self.layer3.call(z)
        z = self.layer4.call(z)
        return self.layer5.call(z)
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


class HiddenModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(2, 5)
        self.layer2 = NeuralLayer(5, 4)
        self.layer3 = NeuralLayer(4, 1)

        self.layer1.init_weights(0.1)
        self.layer2.init_weights(0.1)
        self.layer3.init_weights(0.1)

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        y_hat = self.layer3.call(z2)

        dLoss = self.layer3.grad_loss(z2, y)
        dLayer3 = self.layer3.grad_layer(z2, y_hat)
        dWeight3 = self.layer3.grad_weight(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)

        dWeight1 = self.layer1.grad_weight(x)

        dWeight1 = self.layer1.grad_weight(x)

        # print("Weight: ", dWeight1)

        dW3 = dLoss * dWeight3
        dW2 = np.dot(dLoss * dLayer3, dWeight2)
        dW1 = np.dot(np.dot(dLoss * dLayer3, dLayer2), dWeight1)

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)

        return dW3, dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        z = self.layer2.call(z)
        return self.layer3.call(z)


class DeepModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(2, 4)
        self.layer2 = NeuralLayer(4, 4)
        self.layer3 = NeuralLayer(4, 3)
        self.layer4 = NeuralLayer(3, 1)


        self.layer1.init_weights(0.1)
        self.layer2.init_weights(0.1)
        self.layer3.init_weights(0.1)
        self.layer4.init_weights(0.1)

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        z3 = self.layer3.call(z2)
        y_hat = self.layer4.call(z3)

        dLoss = self.layer4.grad_loss(z3, y)
        dLayer4 = self.layer4.grad_layer(z3, y_hat)
        dWeight4 = self.layer4.grad_weight(z3)

        dLayer3 = self.layer3.grad_layer(z2, z3)
        dWeight3 = self.layer3.grad_weight(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)

        dWeight1 = self.layer1.grad_weight(x)

        dWeight1 = self.layer1.grad_weight(x)

        # print("Weight: ", dWeight1)
        dW4 = dLoss * dWeight4
        dW3 = np.dot(dLoss * dLayer4, dWeight3)
        dW2 = np.dot(np.dot(dLoss * dLayer4, dLayer3), dWeight2)
        dW1 = np.dot(np.dot(np.dot(dLoss * dLayer4, dLayer3), dLayer2), dWeight1)

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)
        self.layer4.weights -= self.training_rate * dW4
        self.layer4.weights = np.clip(self.layer4.weights, 0.0, 1.0)

        return dW4, dW3, dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        z = self.layer2.call(z)
        z = self.layer3.call(z)
        return self.layer4.call(z)



class DeepDeepModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(2, 5)
        self.layer2 = NeuralLayer(5, 4)
        self.layer3 = NeuralLayer(4, 3)
        self.layer4 = NeuralLayer(3, 2)
        self.layer5 = NeuralLayer(2, 1)


        self.layer1.init_weights(0.1)
        self.layer2.init_weights(0.1)
        self.layer3.init_weights(0.1)
        self.layer4.init_weights(0.1)
        self.layer5.init_weights(0.1)

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        z3 = self.layer3.call(z2)
        z4 = self.layer4.call(z3)
        y_hat = self.layer5.call(z4)

        dLoss = self.layer5.grad_loss(z4, y)
        dLayer5 = self.layer5.grad_layer(z4, y_hat)
        dWeight5 = self.layer5.grad_weight(z4)

        dLayer4 = self.layer4.grad_layer(z3, z4)
        dWeight4 = self.layer4.grad_weight(z3)

        dLayer3 = self.layer3.grad_layer(z2, z3)
        dWeight3 = self.layer3.grad_weight(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)

        dWeight1 = self.layer1.grad_weight(x)

        dWeight1 = self.layer1.grad_weight(x)

        # print("Weight: ", dWeight1)
        dW5 = dLoss * dWeight5
        dW4 = np.dot(dLoss * dLayer5, dWeight4)
        dW3 = np.dot(np.dot(dLoss * dLayer5, dLayer4), dWeight3)
        dW2 = np.dot(np.dot(np.dot(dLoss * dLayer5, dLayer4), dLayer3), dWeight2)
        dW1 = np.dot(np.dot(np.dot(np.dot(dLoss * dLayer5, dLayer4), dLayer3), dLayer2), dWeight1)

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)
        self.layer4.weights -= self.training_rate * dW4
        self.layer4.weights = np.clip(self.layer4.weights, 0.0, 1.0)
        self.layer5.weights -= self.training_rate * dW5
        self.layer5.weights = np.clip(self.layer5.weights, 0.0, 1.0)

        return dW4, dW3, dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        z = self.layer2.call(z)
        z = self.layer3.call(z)
        z = self.layer4.call(z)
        return self.layer5.call(z)
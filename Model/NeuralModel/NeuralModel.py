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
        
        dLoss = self.layer2.grad_loss(y_hat, y)
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

        dLoss = self.layer3.grad_loss(y_hat, y)
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

        dLoss = self.layer4.grad_loss(y_hat, y)
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

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

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

        # print("Weight: ", dWeight1)
        dW5 = dLoss * dWeight5
        dW4 = np.dot(dLoss * dLayer5, dWeight4)
        dW3 = np.dot(np.dot(dLoss * dLayer5, dLayer4), dWeight3)
        dW2 = np.dot(np.dot(np.dot(dLoss * dLayer5, dLayer4), dLayer3), dWeight2)
        dW1 = np.dot(np.dot(np.dot(np.dot(dLoss * dLayer5, dLayer4), dLayer3), dLayer2), dWeight1)

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


class HiddenModelRegular:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

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

        dLoss = self.layer3.grad_loss(y_hat, y)
        dLayer3 = self.layer3.grad_layer(z2, y_hat)
        dWeight3 = self.layer3.grad_weight(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)

        dWeight1 = self.layer1.grad_weight(x)

        # print("Weight: ", dWeight1)

        dW3 = dLoss * dWeight3
        dW2 = (dLoss @ dLayer3).T * dWeight2
        dW1 = ((dLoss @ dLayer3) @ dLayer2).T * dWeight1

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights -= self.regularisation * self.layer1.weights
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights -= self.regularisation * self.layer2.weights
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights -= self.regularisation * self.layer3.weights
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)

        return dW3, dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        z = self.layer2.call(z)
        return self.layer3.call(z)


class SimpleModelRegular:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

        self.layer1 = NeuralLayer(3, 3)
        self.layer2 = NeuralLayer(3, 1)

        self.layer1.relu = True
        self.layer2.relu = True
        self.layer2.crossentropy = True

        self.layer1.init_weights(0.1)
        self.layer2.init_weights(0.1)

    def grad(self, x, y):
        z = self.layer1.call(x)
        y_hat = self.layer2.call(z)
        
        dLoss = self.layer2.grad_loss(y_hat, y)
        dLayer2 = self.layer2.grad_layer(z, y_hat)
        dWeight2 = self.layer2.grad_weight(z)

        dWeight1 = self.layer1.grad_weight(x)

        dW2 = dLoss * dWeight2
        dW1 = (dLoss @ dLayer2).T * dWeight1

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights -= self.regularisation * self.layer1.weights
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights -= self.regularisation * self.layer2.weights
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)

        return dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        return self.layer2.call(z)

class Perceptron:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

        self.layer = NeuralLayer(2, 1)
        self.layer.init_weights(0.1)

    def grad(self, x, y):
        y_hat = self.layer.call(x)

        dLoss = self.layer.grad_loss(y_hat, y)
        dLayer = self.layer.grad_layer(x, y_hat)
        dWeight = self.layer.grad_weight(x)

        dW = dLoss * dWeight

        self.layer.weights -= self.training_rate * dW
        self.layer.weights -= self.regularisation * self.layer.weights
        self.layer.weights = np.clip(self.layer.weights, 0.0, 1.0)

        return dW

    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        return self.layer.call(x)


class SimpleModelRelu:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

        self.layer1 = NeuralLayer(3, 3)
        self.layer2 = NeuralLayer(3, 1)

        self.layer1.relu = True
        self.layer2.relu = True
        self.layer2.crossentropy = True

        self.layer1.init_weights(0.2)
        self.layer2.init_weights(0.2)

    def grad(self, x, y):
        z = self.layer1.call(x)
        y_hat = self.layer2.call(z)
        
        dLoss = self.layer2.grad_loss(y_hat, y)
        dLayer2 = self.layer2.grad_layer(z, y_hat)
        dWeight2 = self.layer2.grad_weight(z)

        dWeight1 = self.layer1.grad_weight(x)

        dW2 = dLoss.T * dWeight2
        dW1 = np.dot(dLoss, dLayer2).T * dWeight1

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights -= self.regularisation * self.layer1.weights
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights -= self.regularisation * self.layer2.weights
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)

        return dW2, dW1
    
    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        z = self.layer1.call(x)
        return self.layer2.call(z)

class Perceptron:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

        self.layer = NeuralLayer(2, 1)
        self.layer.init_weights(0.1)

    def grad(self, x, y):
        y_hat = self.layer.call(x)

        dLoss = self.layer.grad_loss(y_hat, y)
        dLayer = self.layer.grad_layer(x, y_hat)
        dWeight = self.layer.grad_weight(x)

        dW = dLoss * dWeight

        self.layer.weights -= self.training_rate * dW
        self.layer.weights -= self.regularisation * self.layer.weights
        self.layer.weights = np.clip(self.layer.weights, 0.0, 1.0)

        return dW

    def loss(self, x, y):
        y_hat = self.call(x)

        return (y - y_hat) ** 2

    def call(self, x):
        return self.layer.call(x)




class TestModel:

    def __init__(self, R):
        self.training_rate = R

        self.layer1 = NeuralLayer(3, 3)
        self.layer2 = NeuralLayer(3, 2)

        self.layer1.init_weights(0.1)
        self.layer2.init_weights(0.1)

        self.layer1.no_activation = True
        self.layer2.no_activation = True

    def grad(self, x, y):
        z = self.layer1.call(x)
        y_hat = self.layer2.call(z)
        
        dLoss = self.layer2.grad_loss(y_hat, y)
        dLayer2 = self.layer2.grad_layer(z, y_hat)
        dWeight2 = self.layer2.grad_weight(z)

        dWeight1 = self.layer1.grad_weight(x)

        # print("dLoss: ", dLoss.shape)
        # print("dLayer2: ", dLayer2.shape)
        # print("dWeight2: ", dWeight2.shape)

        # print("dWeight1: ", dWeight1.shape)

        print("dWeight1: {}".format(dWeight1))

        dW2 = dLoss.T * dWeight2
        dW1 = np.dot(dLoss, dLayer2) * dWeight1

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


def model_test():
    model = TestModel(0.25)

    x = np.random.randint(0, 10, size=3) / 10
    y_hat = np.random.randint(0, 10, size=2) / 10
    w1 = np.random.randint(0, 10, size=(3, 3)) / 10
    w2 = np.random.randint(0, 10, size=(2, 3)) / 10

    model.layer1.weights = w1
    model.layer2.weights = w2

    dw1 = np.zeros((3, 3))
    dw2 = np.zeros((2, 3))

    # call between layers
    z = np.zeros((3))
    z[0] = 1 - (1 - (x[0] * w1[0,0])) * (1 - (x[1] * w1[0,1])) * (1 - (x[2] * w1[0,2]))
    z[1] = 1 - (1 - (x[0] * w1[1,0])) * (1 - (x[1] * w1[1,1])) * (1 - (x[2] * w1[1,2]))
    z[2] = 1 - (1 - (x[0] * w1[2,0])) * (1 - (x[1] * w1[2,1])) * (1 - (x[2] * w1[2,2]))

    y = np.zeros((2))
    y[0] = 1 - ((1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])))
    y[1] = 1 - ((1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])))

    # layer 1 values
    dw1[0,0] = x[0] * (1 - (x[1] * w1[0,1])) * (1 - (x[2] * w1[0,2])) * -2 * ((w2[0,0] * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - y[0])) + (w2[1,0] * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - y[1])))
    dw1[0,1] = x[1] * (1 - (x[0] * w1[0,0])) * (1 - (x[2] * w1[0,2])) * -2 * ((w2[0,1] * (1 - (z[0] * w2[0,0])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - y[0])) + (w2[1,1] * (1 - (z[0] * w2[1,0])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - y[1])))
    dw1[0,2] = x[2] * (1 - (x[0] * w1[0,0])) * (1 - (x[1] * w1[0,1])) * -2 * ((w2[0,2] * (1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (y_hat[0] - y[0])) + (w2[1,2] * (1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (y_hat[1] - y[1])))
    dw1[1,0] = x[0] * (1 - (x[1] * w1[1,1])) * (1 - (x[2] * w1[1,2])) * -2 * ((w2[0,0] * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - y[0])) + (w2[1,0] * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - y[1])))
    dw1[1,1] = x[1] * (1 - (x[0] * w1[1,0])) * (1 - (x[2] * w1[1,2])) * -2 * ((w2[0,1] * (1 - (z[0] * w2[0,0])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - y[0])) + (w2[1,1] * (1 - (z[0] * w2[1,0])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - y[1])))
    dw1[1,2] = x[2] * (1 - (x[0] * w1[1,0])) * (1 - (x[1] * w1[1,1])) * -2 * ((w2[0,2] * (1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (y_hat[0] - y[0])) + (w2[1,2] * (1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (y_hat[1] - y[1])))
    dw1[2,0] = x[0] * (1 - (x[1] * w1[2,1])) * (1 - (x[2] * w1[2,2])) * -2 * ((w2[0,0] * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - y[0])) + (w2[1,0] * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - y[1])))
    dw1[2,1] = x[1] * (1 - (x[0] * w1[2,0])) * (1 - (x[2] * w1[2,2])) * -2 * ((w2[0,1] * (1 - (z[0] * w2[0,0])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - y[0])) + (w2[1,1] * (1 - (z[0] * w2[1,0])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - y[1])))
    dw1[2,2] = x[2] * (1 - (x[0] * w1[2,0])) * (1 - (x[1] * w1[2,1])) * -2 * ((w2[0,2] * (1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (y_hat[0] - y[0])) + (w2[1,2] * (1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (y_hat[1] - y[1])))


    # layer 2 values
    dw2[0,0] = -2 * z[0] * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - 1 + (1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])))
    dw2[0,1] = -2 * z[1] * (1 - (z[1] * w2[0,0])) * (1 - (z[2] * w2[0,2])) * (y_hat[0] - 1 + (1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])))
    dw2[0,2] = -2 * z[2] * (1 - (z[1] * w2[0,0])) * (1 - (z[2] * w2[0,1])) * (y_hat[0] - 1 + (1 - (z[0] * w2[0,0])) * (1 - (z[1] * w2[0,1])) * (1 - (z[2] * w2[0,2])))
    dw2[1,0] = -2 * z[0] * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - 1 + (1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])))
    dw2[1,1] = -2 * z[1] * (1 - (z[1] * w2[1,0])) * (1 - (z[2] * w2[1,2])) * (y_hat[1] - 1 + (1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])))
    dw2[1,2] = -2 * z[2] * (1 - (z[1] * w2[1,0])) * (1 - (z[2] * w2[1,1])) * (y_hat[1] - 1 + (1 - (z[0] * w2[1,0])) * (1 - (z[1] * w2[1,1])) * (1 - (z[2] * w2[1,2])))

    dw2_hat, dw1_hat = model.grad(x, y_hat)

    print("\n\nLayer 1 Gradient")
    print("Expected: {}".format(dw1))
    print("Got:      {}".format(dw1_hat))

    print("\n\nLayer 2 Gradient")
    print("Expected: {}".format(dw2))
    print("Got:      {}".format(dw2_hat))




if __name__ == "__main__":
    model_test()
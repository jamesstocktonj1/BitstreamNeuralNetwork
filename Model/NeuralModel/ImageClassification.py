import tensorflow as tf
import numpy as np

from NeuralLayer import NeuralLayer



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape((60000, 28*28))
x_test = x_test.reshape((10000, 28*28))

y_train_data = np.zeros((60000, 10))
for i in range(60000):
    y_train_data[i,y_train[i]] = 1

y_test_data = np.zeros((10000, 10))
for i in range(10000):
    y_test_data[i,y_test[i]] = 1



class RegularisedModel:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

        self.layer1 = NeuralLayer(28*28, 128)
        self.layer2 = NeuralLayer(128, 127)
        self.layer3 = NeuralLayer(127, 10)

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

        # print("Weight: ", dWeight1)

        print(dLoss.shape)
        print(dLayer3.shape)
        print(dWeight3.shape)
        print(dLayer2.shape)
        print(dWeight2.shape)
        print(dWeight1.shape)

        print(self.layer3.weights.shape)
        print(self.layer2.weights.shape)
        print(self.layer1.weights.shape)


        dW3 = dLoss * dWeight3
        dW2 = np.dot(dLoss * dLayer3, dWeight2)
        dW1 = np.dot(np.dot(dLoss * dLayer3, dLayer2), dWeight1)

        self.layer1.weights -= self.training_rate * dW1
        self.layer1.weights -= self.regularisation * self.layer1.weights
        self.layer1.weights = np.clip(self.layer1.weights, 0.0, 1.0)
        self.layer2.weights -= self.training_rate * dW2
        self.layer2.weights -= self.regularisation * self.layer2.weights
        self.layer2.weights = np.clip(self.layer2.weights, 0.0, 1.0)
        self.layer3.weights -= self.training_rate * dW3
        self.layer3.weights -= self.regularisation * self.layer3.weights
        self.layer3.weights = np.clip(self.layer3.weights, 0.0, 1.0)

    def loss(self, x, y):
        y_hat = self.call(x)

        return np.sum((y - y_hat) ** 2)

    def call(self, x):
        z = self.layer1.call(x)
        z = self.layer2.call(z)
        return self.layer3.call(z)


model = RegularisedModel(0.025, 0.00007)


def training_loop():
    
    lossEpoch = []
    for e in range(25):

        # stochastic gradient descent
        randPoints = np.random.choice(np.arange(x_train.shape[0]), 100)
        for rxy in randPoints:
            print(x_train[rxy])
            model.grad(x_train[rxy], y_train_data[rxy])

        # calculate model loss
        modelLoss = 0
        for rxy in range(x_train.shape[0]):
            modelLoss += model.loss(x_train[rxy], y_train_data[rxy])

        print("Loss: {}".format(modelLoss))
    


def testing_loop():

    correct = np.zeros(y_test.shape[0])


if __name__ == "__main__":
    training_loop()
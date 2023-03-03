import numpy as np
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from NeuralLayer import NeuralLayer



# PARAMETERS
LEARNING_RATE = 0.125
NORMALISATION = 0.000007
EPOCH_COUNT = 2500
BATCH_SIZE = 45
WEIGHTS_PARAMETER = 0.25


iris_data = load_iris()

x_data = iris_data.data
x_data = x_data/np.max(x_data, axis=0)
y_data = iris_data.target.reshape(-1, 1)

x_train, x_test, y_train_index, y_test_index = train_test_split(x_data, y_data, test_size=0.2)


y_train = np.zeros((x_train.shape[0], 3))
for i, target in enumerate(y_train_index):
    y_train[i][target] = 1

y_test = np.zeros((x_test.shape[0], 3))
for i, target in enumerate(y_test_index):
    y_test[i][target] = 1


class IrisModel:

    def __init__(self, R, L):
        self.training_rate = R
        self.regularisation = L

        self.layer1 = NeuralLayer(4, 8)
        self.layer2 = NeuralLayer(8, 6)
        self.layer3 = NeuralLayer(6, 3)

        self.layer1.init_xavier(WEIGHTS_PARAMETER)
        self.layer2.init_xavier(WEIGHTS_PARAMETER)
        self.layer3.init_xavier(WEIGHTS_PARAMETER)

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

        dW3 = dLoss.T * dWeight3
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


model = IrisModel(LEARNING_RATE, NORMALISATION)


def testing_loop():
    modelLoss = 0
    correctCount = 0

    for rxy in range(x_test.shape[0]):
        modelLoss += model.loss(x_test[rxy], y_train[rxy])

        y_hat = model.call(x_test[rxy])
        if y_hat.argmax() == y_test[rxy].argmax():
            correctCount += 1

    modelLoss = modelLoss.sum() / x_test.shape[0]

    print("Testing Loss: {}".format(modelLoss))
    print("Correct Values: {}/{}".format(correctCount, x_test.shape[0]))


def training_loop():

    lossEpoch = []

    for e in range(EPOCH_COUNT):

        randPoints = np.random.choice(np.arange(x_train.shape[0]), BATCH_SIZE)
        for rxy in randPoints:
            model.grad(x_train[rxy], y_train[rxy])

        modelLoss = 0
        for rxy in range(x_train.shape[0]):
            modelLoss += model.loss(x_train[rxy], y_train[rxy])
        
        modelLoss = modelLoss.sum() / x_train.shape[0]
        lossEpoch.append(modelLoss)

        print("\nEpoch {}, Loss: {}".format(e, modelLoss))
        testing_loop()





if __name__ == "__main__":
    training_loop()
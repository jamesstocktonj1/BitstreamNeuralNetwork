from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import json, sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from NeuralLayer import NeuralLayer, NeuralLayerBias



# PARAMETERS
LEARNING_RATE = 0.0125
NORMALISATION = 0.00005
EPOCH_COUNT = 150
BATCH_SIZE = 0.8
WEIGHTS_PARAMETER = 0.5
CORRECT_THRESHOLD = 0.01

X = 50

x_data = np.arange(0, 1, 1/X)
y_data = (0.4 * np.sin((4 * x_data) - 1)) + (0.12 * np.cos(15 * x_data)) + np.exp((-5 * x_data) - 1) + 0.4

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)



def plot_function(x_data, y_data, y_hat):
    plt.figure()

    plt.plot(x_data, y_data)
    plt.plot(x_data, y_hat)

    plt.plot(x_train, y_train, 'bo')
    plt.plot(x_test, y_test, 'rx')

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig("images/function/function.png")
    plt.close()

def plot_model_progress(trainLoss, trainCorrect):
    plt.figure()

    fig1 = plt.subplot(2, 1, 1)
    fig2 = plt.subplot(2, 1, 2)

    fig1.plot(np.arange(len(trainLoss)), trainLoss)
    fig2.plot(np.arange(len(trainCorrect)), trainCorrect)

    fig1.set_ylabel("Training Loss")
    fig2.set_ylabel("Correct Train Points (%)")
    fig2.set_xlabel("Epochs")

    plt.savefig("images/function/function_epochs.png")
    plt.close()

def loss_stats():

    testLoss = 0
    testCorrect = 0
    trainLoss = 0
    trainCorrect = 0

    for rxy in range(x_train.shape[0]):
        trainLoss += model.loss(x_train[rxy], y_train[rxy])

        y_hat = model.call(x_train[rxy])
        if np.abs(y_hat - y_train[rxy]) < CORRECT_THRESHOLD:
            trainCorrect += 1

    y_pred = np.zeros(x_test.shape[0])
    for rxy in range(x_test.shape[0]):
        testLoss += model.loss(x_test[rxy], y_test[rxy])

        y_hat = model.call(x_test[rxy])
        y_pred[rxy] = y_hat
        if np.abs(y_hat - y_test[rxy]) < CORRECT_THRESHOLD:
            testCorrect += 1

    print("Network Stats:")
    print("Training Loss: {}\t Training Accuracy: {}".format(trainLoss.sum() / x_train.shape[0], (trainCorrect / x_train.shape[0]) * 100))
    print("Testing Loss:  {}\t Testing Accuracy:  {}".format(testLoss.sum() / x_test.shape[0], (testCorrect / x_test.shape[0]) * 100))


class FunctionModel:

    def __init__(self, R, L, U):
        self.training_rate = R
        self.regularisation = L

        self.crossentropy = False

        self.layer1 = NeuralLayerBias(1, 65)
        self.layer2 = NeuralLayerBias(65, 65)
        self.layer3 = NeuralLayerBias(65, 64)
        self.layer4 = NeuralLayerBias(64, 1)

        self.layer1.init_xavier(U)
        self.layer2.init_xavier(U)
        self.layer3.init_xavier(U)
        self.layer4.init_xavier(U)

        self.layer1.no_activation = True
        self.layer2.no_activation = True
        self.layer3.no_activation = True
        self.layer4.no_activation = True
        self.layer4.crossentropy = self.crossentropy

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        z3 = self.layer3.call(z2)
        y_hat = self.layer4.call(z3)

        dLoss = self.layer4.grad_loss(y_hat, y)
        dLayer4 = self.layer4.grad_layer(z3, y_hat)
        dWeight4 = self.layer4.grad_weight(z3)
        dBias4 = self.layer4.grad_bias(z3)

        dLayer3 = self.layer3.grad_layer(z2, z3)
        dWeight3 = self.layer3.grad_weight(z2)
        dBias3 = self.layer3.grad_bias(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)
        dBias2 = self.layer2.grad_bias(z1)

        dWeight1 = self.layer1.grad_weight(x)
        dBias1 = self.layer1.grad_bias(x)

        dW4 = dLoss.T * dWeight4
        dW3 = np.dot(dLoss, dLayer4).T * dWeight3
        dW2 = np.dot(np.dot(dLoss, dLayer4), dLayer3).T * dWeight2
        dW1 = np.dot(np.dot(np.dot(dLoss, dLayer4), dLayer3), dLayer2).T * dWeight1

        dB4 = dLoss.T * dBias4
        dB3 = np.dot(dLoss, dLayer4).T * dBias3
        dB2 = np.dot(np.dot(dLoss, dLayer4), dLayer3).T * dBias2
        dB1 = np.dot(np.dot(np.dot(dLoss, dLayer4), dLayer3), dLayer2).T * dBias1

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

        self.layer4.bias -= self.training_rate * dB4
        self.layer4.bias -= self.regularisation * self.layer4.bias
        self.layer4.bias = np.clip(self.layer4.bias, 0.0, 1.0)
        self.layer3.bias -= self.training_rate * dB3
        self.layer3.bias -= self.regularisation * self.layer3.bias
        self.layer3.bias = np.clip(self.layer3.bias, 0.0, 1.0)
        self.layer2.bias -= self.training_rate * dB2
        self.layer2.bias -= self.regularisation * self.layer2.bias
        self.layer2.bias = np.clip(self.layer2.bias, 0.0, 1.0)
        self.layer1.bias -= self.training_rate * dB1
        self.layer1.bias -= self.regularisation * self.layer1.bias
        self.layer1.bias = np.clip(self.layer1.bias, 0.0, 1.0)

        return dW4, dW3, dW2, dW1
    
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
        return self.layer4.call(z)


class FunctionBigModel:

    def __init__(self, R, L, U):
        self.training_rate = R
        self.regularisation = L

        self.crossentropy = False

        self.layer1 = NeuralLayerBias(1, 5)
        self.layer2 = NeuralLayerBias(5, 25)
        self.layer3 = NeuralLayerBias(25, 20)
        self.layer4 = NeuralLayerBias(20, 16)
        self.layer5 = NeuralLayerBias(16, 15)
        self.layer6 = NeuralLayerBias(15, 4)
        self.layer7 = NeuralLayerBias(4, 1)

        self.layer1.init_uniform(U)
        self.layer2.init_uniform(U)
        self.layer3.init_uniform(U)
        self.layer4.init_uniform(U)
        self.layer5.init_uniform(U)
        self.layer6.init_uniform(U)
        self.layer7.init_uniform(U)

        self.layer1.no_activation = True
        self.layer2.no_activation = True
        self.layer3.no_activation = True
        self.layer4.no_activation = True
        self.layer5.no_activation = True
        self.layer6.no_activation = True
        self.layer7.no_activation = True
        self.layer7.crossentropy = self.crossentropy

    def grad(self, x, y):
        z1 = self.layer1.call(x)
        z2 = self.layer2.call(z1)
        z3 = self.layer3.call(z2)
        z4 = self.layer4.call(z3)
        z5 = self.layer5.call(z4)
        z6 = self.layer6.call(z5)
        y_hat = self.layer7.call(z6)

        dLoss = self.layer7.grad_loss(y_hat, y)
        dLayer7 = self.layer7.grad_layer(z6, y_hat)
        dWeight7 = self.layer7.grad_weight(z6)
        dBias7 = self.layer7.grad_bias(z6)

        dLayer6 = self.layer6.grad_layer(z5, z6)
        dWeight6 = self.layer6.grad_weight(z5)
        dBias6 = self.layer6.grad_bias(z5)

        dLayer5 = self.layer5.grad_layer(z4, z5)
        dWeight5 = self.layer5.grad_weight(z4)
        dBias5 = self.layer5.grad_bias(z4)


        dLayer4 = self.layer4.grad_layer(z3, z4)
        dWeight4 = self.layer4.grad_weight(z3)
        dBias4 = self.layer4.grad_bias(z3)

        dLayer3 = self.layer3.grad_layer(z2, z3)
        dWeight3 = self.layer3.grad_weight(z2)
        dBias3 = self.layer3.grad_bias(z2)

        dLayer2 = self.layer2.grad_layer(z1, z2)
        dWeight2 = self.layer2.grad_weight(z1)
        dBias2 = self.layer2.grad_bias(z1)

        dWeight1 = self.layer1.grad_weight(x)
        dBias1 = self.layer1.grad_bias(x)

        dW7 = dLoss.T * dWeight7
        dW6 = np.dot(dLoss, dLayer7).T * dWeight6
        dW5 = np.dot(np.dot(dLoss, dLayer7), dLayer6).T * dWeight5
        dW4 = np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5).T * dWeight4
        dW3 = np.dot(np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5), dLayer4).T * dWeight3
        dW2 = np.dot(np.dot(np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5), dLayer4), dLayer3).T * dWeight2
        dW1 = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5), dLayer4), dLayer3), dLayer2).T * dWeight1

        dB7 = dLoss.T * dBias7
        dB6 = np.dot(dLoss, dLayer7).T * dBias6
        dB5 = np.dot(np.dot(dLoss, dLayer7), dLayer6).T * dBias5
        dB4 = np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5).T * dBias4
        dB3 = np.dot(np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5), dLayer4).T * dBias3
        dB2 = np.dot(np.dot(np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5), dLayer4), dLayer3).T * dBias2
        dB1 = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(dLoss, dLayer7), dLayer6), dLayer5), dLayer4), dLayer3), dLayer2).T * dBias1

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
        self.layer6.weights -= self.training_rate * dW6
        self.layer6.weights -= self.regularisation * self.layer6.weights
        self.layer6.weights = np.clip(self.layer6.weights, 0.0, 1.0)
        self.layer7.weights -= self.training_rate * dW7
        self.layer7.weights -= self.regularisation * self.layer7.weights
        self.layer7.weights = np.clip(self.layer7.weights, 0.0, 1.0)

        self.layer7.bias -= self.training_rate * dB7
        self.layer7.bias -= self.regularisation * self.layer7.bias
        self.layer7.bias = np.clip(self.layer7.bias, 0.0, 1.0)
        self.layer6.bias -= self.training_rate * dB6
        self.layer6.bias -= self.regularisation * self.layer6.bias
        self.layer6.bias = np.clip(self.layer6.bias, 0.0, 1.0)
        self.layer5.bias -= self.training_rate * dB5
        self.layer5.bias -= self.regularisation * self.layer5.bias
        self.layer5.bias = np.clip(self.layer5.bias, 0.0, 1.0)
        self.layer4.bias -= self.training_rate * dB4
        self.layer4.bias -= self.regularisation * self.layer4.bias
        self.layer4.bias = np.clip(self.layer4.bias, 0.0, 1.0)
        self.layer3.bias -= self.training_rate * dB3
        self.layer3.bias -= self.regularisation * self.layer3.bias
        self.layer3.bias = np.clip(self.layer3.bias, 0.0, 1.0)
        self.layer2.bias -= self.training_rate * dB2
        self.layer2.bias -= self.regularisation * self.layer2.bias
        self.layer2.bias = np.clip(self.layer2.bias, 0.0, 1.0)
        self.layer1.bias -= self.training_rate * dB1
        self.layer1.bias -= self.regularisation * self.layer1.bias
        self.layer1.bias = np.clip(self.layer1.bias, 0.0, 1.0)

        return dW4, dW3, dW2, dW1
    
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
        z = self.layer5.call(z)
        z = self.layer6.call(z)
        return self.layer7.call(z)


model = FunctionModel(LEARNING_RATE, NORMALISATION, WEIGHTS_PARAMETER)

def testing_loop():

    pass



def training_loop():

    trainLoss = []
    trainEpoch = []

    correctArr = np.zeros(x_train.shape[0])

    for e in range(EPOCH_COUNT):

        preLoss = 0
        postLoss = 0

        randPoints = np.random.choice(np.where(correctArr == 0)[0], int(BATCH_SIZE * x_train.shape[0]))
        # randPoints = np.random.choice(np.arange(x_train.shape[0]), int(BATCH_SIZE * x_train.shape[0]))
        # randPoints = np.arange(x_train.shape[0])
        for rxy in randPoints:
            preLoss += model.loss(x_train[rxy], y_train[rxy])

        for rxy in randPoints:
            model.grad(x_train[rxy], y_train[rxy])

        for rxy in randPoints:
            postLoss += model.loss(x_train[rxy], y_train[rxy])

        modelLoss = 0
        correctCount = 0
        for rxy in range(x_train.shape[0]):
            modelLoss += model.loss(x_train[rxy], y_train[rxy])

            y_hat = model.call(x_train[rxy])
            if np.abs(y_hat - y_train[rxy]) < CORRECT_THRESHOLD:
                correctArr[rxy] = 1
                correctCount += 1
            else:
                correctArr[rxy] = 0
                
        
        modelLoss = modelLoss.sum() / x_train.shape[0]

        print("\nEpoch {}:".format(e))
        print("Training Loss: {}\tCorrect Values: {}/{}".format(modelLoss, correctCount, x_train.shape[0]))

        trainLoss.append(modelLoss)
        trainEpoch.append((correctCount / x_train.shape[0]) * 100)

        model.training_rate = LEARNING_RATE * modelLoss

    y_hat = []
    for rxy in x_data:
        y_hat.append(model.call(rxy))

    plot_function(x_data, y_data, y_hat)
    plot_model_progress(trainLoss, trainEpoch)


if __name__ == "__main__":
    training_loop()

    loss_stats()
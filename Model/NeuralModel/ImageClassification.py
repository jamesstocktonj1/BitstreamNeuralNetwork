import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json

from NeuralLayer import NeuralLayer



# PARAMETERS
LEARNING_RATE = 1.25
NORMALISATION = 0.00007
EPOCH_COUNT = 2500
BATCH_SIZE = 2
WEIGHTS_PARAMETER = 0.25


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

    def init_weights(self, u):
        self.layer1.init_xavier(u)
        self.layer2.init_xavier(u)
        self.layer3.init_xavier(u)

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


        dW3 = dLoss * dWeight3
        dW2 = (dLoss.T @ dLayer3).T * dWeight2
        dW1 = ((dLoss.T @ dLayer3) @ dLayer2).T * dWeight1


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


model = RegularisedModel(LEARNING_RATE, NORMALISATION)


status = {
    "losses": [],
    "epochs": []
}

modelInfo = {}

def save_json(name):
    with open("data/image_{}.json".format(name), "w") as f:
        json_obj = json.dumps(status, indent=4)
        f.write(json_obj)


def save_parameters():
    
    modelInfo["name"] = "Image Classification"
    modelInfo["learning_rate"] = LEARNING_RATE
    modelInfo["normalisation_rate"] = NORMALISATION
    modelInfo["epoch_count"] = EPOCH_COUNT
    modelInfo["batch_size"] = BATCH_SIZE
    modelInfo["xavier_parameter"] = WEIGHTS_PARAMETER

    status["info"] = modelInfo

    save_json("info")

def save_status(modelLoss, epoch):

    # add epoch information to file
    epochStatus = {}
    epochStatus["info"] = "Epoch {}".format(epoch)
    epochStatus["layer1"] = model.layer1.weights.tolist()
    epochStatus["layer2"] = model.layer2.weights.tolist()
    epochStatus["layer3"] = model.layer3.weights.tolist()
    epochStatus["loss"] = modelLoss

    status["losses"].append(modelLoss)

    if len(status["epochs"]) > 100:
        status["epochs"] = []
    status["epochs"].append(epochStatus)

    save_json(epoch // 100)

    # update losses in info file
    modelInfo["losses"] = modelLoss
    save_json("info")


def plot_loss_epoch(loss):
    plt.figure()

    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(np.arange(len(loss)), loss)

    plt.savefig("images/image_epoch_loss.png")
    plt.close()


def train_function(index):
    return model.grad(x_train[index], y_train_data[index])

train_point = np.vectorize(train_function)

def loss_function(index):
    return model.loss(x_train[index], y_train_data[index])

get_loss = np.vectorize(loss_function)

def training_loop():

    model.init_weights(WEIGHTS_PARAMETER)
    
    lossEpoch = []
    for e in range(EPOCH_COUNT):

        # stochastic gradient descent
        randPoints = np.random.choice(np.arange(x_train.shape[0]), BATCH_SIZE)
        # for rxy in randPoints:
        #     # print(x_train[rxy])
        #     model.grad(x_train[rxy], y_train_data[rxy])

        train_point(randPoints)

        # calculate model loss
        # modelLoss = 0
        # for rxy in range(x_train.shape[0]):
        #     modelLoss += model.loss(x_train[rxy], y_train_data[rxy])

        # modelLoss = get_loss(randPoints).sum()
        modelLoss = get_loss(np.arange(x_train.shape[0])).sum()
        modelLoss = modelLoss / x_train.shape[0]

        print("Epoch {}, Loss: {}".format(e, modelLoss))
        lossEpoch.append(modelLoss)

        testing_loop()

        save_status(modelLoss, e)
    
    plot_loss_epoch(lossEpoch)


def testing_loop():

    y_hat = model.call(x_train[0])
    y = y_train_data[0]

    print("Exp {}".format(y))
    print("Got {}".format(y_hat))

if __name__ == "__main__":
    save_parameters()
    training_loop()

    testing_loop()
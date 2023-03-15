import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from NeuralLayer import NeuralLayer
from IrisModel import IrisModel, IrisModel2



# PARAMETERS
LEARNING_RATE = 0.0375
NORMALISATION = 0
EPOCH_COUNT = 150
BATCH_SIZE = 200
WEIGHTS_PARAMETER = 0.7
CORRECT_THRESHOLD = 0.075


diabetes_data = load_diabetes()

x_data = diabetes_data.data
x_data = (x_data - np.min(x_data,axis=0)) / (np.max(x_data, axis=0) - np.min(x_data, axis=0))
y_data = diabetes_data.target.reshape(-1, 1)
y_data = (y_data - np.min(y_data,axis=0)) / (np.max(y_data, axis=0) - np.min(y_data, axis=0))

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)


class DiabetesModel:

    def __init__(self, R, L, U):
        self.training_rate = R
        self.regularisation = L

        self.crossentropy = False

        self.layer1 = NeuralLayer(10, 16)
        self.layer2 = NeuralLayer(16, 14)
        self.layer3 = NeuralLayer(14, 5)
        self.layer4 = NeuralLayer(5, 1)

        self.layer1.init_weights(U)
        self.layer2.init_weights(U)
        self.layer3.init_weights(U)
        self.layer4.init_weights(U)

        self.layer1.relu = True
        self.layer2.relu = True
        self.layer3.relu = True

        self.layer4.crossentropy = self.crossentropy
        # self.layer4.no_activation = True

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

        dW4 = dLoss.T * dWeight4
        dW3 = np.dot(dLoss, dLayer4).T * dWeight3
        dW2 = np.dot(np.dot(dLoss, dLayer4), dLayer3).T * dWeight2
        dW1 = np.dot(np.dot(np.dot(dLoss, dLayer4), dLayer3), dLayer2).T * dWeight1

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


model = DiabetesModel(LEARNING_RATE, NORMALISATION, WEIGHTS_PARAMETER)


modelInfo = {
    "epoch": []
}

modelStatus = {
    "epoch": [],
    "train_loss": [],
    "testing_loss": []
}


def save_json(name, data):
    with open("data/diabetes/diabetes_{}.json".format(name), "w") as f:
        json_obj = json.dumps(data, indent=4)
        f.write(json_obj)

def save_parameters():
    
    modelInfo["name"] = "Image Classification"
    modelInfo["learning_rate"] = LEARNING_RATE
    modelInfo["normalisation_rate"] = NORMALISATION
    modelInfo["epoch_count"] = EPOCH_COUNT
    modelInfo["batch_size"] = BATCH_SIZE
    modelInfo["xavier_parameter"] = WEIGHTS_PARAMETER

    modelStatus["info"] = modelInfo

    save_json("info", modelInfo)

def save_status(trainLoss, testLoss, trainCorrect, testCorrect, epoch):

    epochStatus = {}
    epochStatus["info"] = "Epoch {}".format(epoch)
    epochStatus["training_loss"] = trainLoss
    epochStatus["training_correct"] = trainCorrect
    epochStatus["testing_loss"] = testLoss
    epochStatus["testing_correct"] = testCorrect

    modelInfo["epoch"].append(epochStatus)


    epochStatus["layer1"] = model.layer1.weights.tolist()
    epochStatus["layer2"] = model.layer2.weights.tolist()
    epochStatus["layer3"] = model.layer3.weights.tolist()

    if len(modelStatus["epoch"]) > 99:
        modelStatus["epoch"] = []

    modelStatus["epoch"].append(epochStatus)
    modelStatus["train_loss"].append(trainLoss)
    modelStatus["testing_loss"].append(testLoss)

    save_json("info", modelInfo)
    save_json(epoch // 100, modelStatus)

def load_model(name):

    with open(name, "r") as f:
        modelData = json.load(f)
    
    print("Loading Model Info: {}".format(modelData["info"]))

    model.layer1.weights = np.array(modelData["layer1"])
    model.layer2.weights = np.array(modelData["layer2"])
    model.layer3.weights = np.array(modelData["layer3"])
    model.layer4.weights = np.array(modelData["layer4"])


def plot_model_progress(trainLoss, testLoss, trainCorrect, testCorrect):
    plt.figure()

    fig1 = plt.subplot(4, 1, 1)
    fig2 = plt.subplot(4, 1, 2)
    fig3 = plt.subplot(4, 1, 3)
    fig4 = plt.subplot(4, 1, 4)

    fig1.plot(np.arange(len(trainLoss)), trainLoss)
    fig2.plot(np.arange(len(testLoss)), testLoss)
    fig3.plot(np.arange(len(trainCorrect)), trainCorrect)
    fig4.plot(np.arange(len(testCorrect)), testCorrect)

    fig1.set_ylabel("Training Loss")
    fig2.set_ylabel("Testing Loss")
    fig3.set_ylabel("Correct Train Points (%)")
    fig4.set_ylabel("Correct Test Points (%)")
    fig4.set_xlabel("Epochs")

    plt.savefig("images/diabetes/diabetes_epochs.png")
    plt.close()

def loss_stats():

    testLoss = 0
    testCorrect = 0
    trainLoss = 0
    trainCorrect = 0

    for rxy in range(x_train.shape[0]):
        trainLoss += model.loss(x_train[rxy], y_train[rxy])

        y_hat = model.call(x_train[rxy])
        if np.abs(y_hat, y_train[rxy]) < CORRECT_THRESHOLD:
            trainCorrect += 1

    for rxy in range(x_test.shape[0]):
        testLoss += model.loss(x_test[rxy], y_test[rxy])

        y_hat = model.call(x_test[rxy])
        if np.abs(y_hat, y_test[rxy]) < CORRECT_THRESHOLD:
            testCorrect += 1


    print("Network Stats:")
    print("Training Loss: {}\t Training Accuracy: {}".format(trainLoss.sum() / x_train.shape[0], (trainCorrect / x_train.shape[0]) * 100))
    print("Testing Loss:  {}\t Testing Accuracy:  {}".format(testLoss.sum() / x_test.shape[0], (testCorrect / x_test.shape[0]) * 100))


def testing_loop():
    modelLoss = 0
    correctCount = 0

    for rxy in range(x_test.shape[0]):
        modelLoss += model.loss(x_test[rxy], y_test[rxy])

        y_hat = model.call(x_test[rxy])
        if np.abs(y_hat - y_test[rxy]) < CORRECT_THRESHOLD:
            correctCount += 1

    modelLoss = modelLoss.sum() / x_test.shape[0]

    print("Testing Loss: {}\tCorrect Values: {}/{}".format(modelLoss, correctCount, x_test.shape[0]))

    return modelLoss, (correctCount / x_test.shape[0]) * 100

def test_confusion():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    y_hat = np.zeros((x_test.shape[0]))

    for rxy in range(x_test.shape[0]):
        y_hat[rxy] = model.call(x_test[rxy]).argmax()

    confMat = confusion_matrix(y_hat, y_test)

    ax.matshow(confMat, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(confMat.shape[0]):
        for j in range(confMat.shape[1]):
            plt.text(x=j, y=i, s=confMat[i, j], va='center', ha='center')

    plt.xlabel("True Prediction")
    plt.ylabel("Model Prediction")

    plt.savefig("images/diabetes/diabetes_confusion.png")
    plt.close()



def predict_loop():

    print("Training Points:")

    for rxy in range(10):
        y_hat = model.call(x_train[rxy])

        print("Expected: {}".format(y_train[rxy]))
        print("Got: {}\n".format(y_hat))


    print("Testing Points")

    for rxy in range(10):
        y_hat = model.call(x_test[rxy])

        print("Expected: {}".format(y_test[rxy]))
        print("Got: {}\n".format(y_hat))


def training_loop():

    trainLoss = []
    testLoss = []
    trainEpoch = []
    testEpoch = []

    for e in range(EPOCH_COUNT):

        randPoints = np.random.choice(np.arange(x_train.shape[0]), BATCH_SIZE)
        for rxy in randPoints:
            model.grad(x_train[rxy], y_train[rxy])

        modelLoss = 0
        correctCount = 0
        for rxy in range(x_train.shape[0]):
            modelLoss += model.loss(x_train[rxy], y_train[rxy])

            y_hat = model.call(x_train[rxy])
            if np.abs(y_hat - y_train[rxy]) < CORRECT_THRESHOLD:
                correctCount += 1
                
        
        modelLoss = modelLoss.sum() / x_train.shape[0]

        print("\nEpoch {}:".format(e))
        print("Training Loss: {}\tCorrect Values: {}/{}".format(modelLoss, correctCount, x_train.shape[0]))
        testingLoss, correctPoints = testing_loop()

        save_status(modelLoss, testingLoss, (correctCount / x_train.shape[0]) * 100, correctPoints, e)

        trainLoss.append(modelLoss)
        testLoss.append(testingLoss)
        trainEpoch.append((correctCount / x_train.shape[0]) * 100)
        testEpoch.append(correctPoints)


    plot_model_progress(trainLoss, testLoss, trainEpoch, testEpoch)




if __name__ == "__main__":

    if False:
        # predict_loop()
        test_confusion()
        loss_stats()
    else:
        save_parameters()
        training_loop()
        predict_loop()
        # test_confusion()
        loss_stats()
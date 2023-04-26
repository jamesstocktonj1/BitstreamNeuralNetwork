import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from NeuralLayer import NeuralLayer
from IrisModel import IrisModel, IrisModel2



# PARAMETERS
LEARNING_RATE = 0.00125
NORMALISATION = 0.0000007
EPOCH_COUNT = 250
BATCH_SIZE = 120
WEIGHTS_PARAMETER = 0.4


iris_data = load_iris()

x_data = iris_data.data
x_data = x_data/np.max(x_data, axis=0)
# x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))
y_data = iris_data.target.reshape(-1, 1)

x_train, x_test, y_train_index, y_test_index = train_test_split(x_data, y_data, test_size=0.2)


y_train = np.zeros((x_train.shape[0], 3))
for i, target in enumerate(y_train_index):
    y_train[i][target] = 1

y_test = np.zeros((x_test.shape[0], 3))
for i, target in enumerate(y_test_index):
    y_test[i][target] = 1


model = IrisModel(LEARNING_RATE, NORMALISATION, WEIGHTS_PARAMETER)


modelInfo = {
    "epoch": []
}

modelStatus = {
    "epoch": [],
    "train_loss": [],
    "testing_loss": []
}


def save_json(name, data):
    with open("data/iris_{}.json".format(name), "w") as f:
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
    epochStatus["layer1b"] = model.layer1.bias.tolist()
    epochStatus["layer2"] = model.layer2.weights.tolist()
    epochStatus["layer2b"] = model.layer2.bias.tolist()
    epochStatus["layer3"] = model.layer3.weights.tolist()
    epochStatus["layer3b"] = model.layer3.bias.tolist()

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

    model.layer1.bias = np.array(modelData["layer1b"])
    model.layer2.bias = np.array(modelData["layer2b"])
    model.layer3.bias = np.array(modelData["layer3b"])


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

    plt.savefig("images/iris_epochs.png")
    plt.close()

def loss_stats():

    testLoss = 0
    testCorrect = 0
    trainLoss = 0
    trainCorrect = 0

    for rxy in range(x_train.shape[0]):
        trainLoss += model.loss(x_train[rxy], y_train[rxy])

        y_hat = model.call(x_train[rxy])
        if y_hat.argmax() == y_train[rxy].argmax():
            trainCorrect += 1

        print(y_hat)

    for rxy in range(x_test.shape[0]):
        testLoss += model.loss(x_test[rxy], y_test[rxy])

        y_hat = model.call(x_test[rxy])
        if y_hat.argmax() == y_test[rxy].argmax():
            testCorrect += 1


    print("Network Stats:")
    print("Training Loss: {}\t Training Accuracy: {}".format(trainLoss.sum() / x_train.shape[0], (trainCorrect / x_train.shape[0]) * 100))
    print("Testing Loss:  {}\t Testing Accuracy:  {}".format(testLoss.sum() / x_test.shape[0], (testCorrect / x_test.shape[0]) * 100))


def testing_loop():
    modelLoss = 0
    correctCount = 0

    for rxy in range(x_test.shape[0]):
        modelLoss += model.loss(x_test[rxy], y_train[rxy])

        y_hat = model.call(x_test[rxy])
        if y_hat.argmax() == y_test[rxy].argmax():
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

    confMat = confusion_matrix(y_hat, y_test_index)

    ax.matshow(confMat, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(confMat.shape[0]):
        for j in range(confMat.shape[1]):
            plt.text(x=j, y=i, s=confMat[i, j], va='center', ha='center')

    plt.xlabel("True Prediction")
    plt.ylabel("Model Prediction")

    plt.savefig("images/iris_confusion.png")
    plt.close()

def train_confusion():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    y_hat = np.zeros((x_train.shape[0]))

    for rxy in range(x_train.shape[0]):
        y_hat[rxy] = model.call(x_train[rxy]).argmax()

    confMat = confusion_matrix(y_hat, y_train_index)

    ax.matshow(confMat, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(confMat.shape[0]):
        for j in range(confMat.shape[1]):
            plt.text(x=j, y=i, s=confMat[i, j], va='center', ha='center')

    plt.xlabel("True Prediction")
    plt.ylabel("Model Prediction")

    plt.savefig("images/iris_train_confusion.png")
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

        correct = np.zeros(x_train.shape[0])
        for rxy in range(x_train.shape[0]):
            y_hat = model.call(x_train[rxy])

            if y_hat.argmax() == y_train[rxy].argmax():
                correct[rxy] = 1

        randPoints = np.random.choice(correct, BATCH_SIZE)
        for rxy in np.where(correct < 1)[0]:
            model.grad(x_train[rxy], y_train[rxy])

        modelLoss = 0
        correctCount = 0
        for rxy in range(x_train.shape[0]):
            modelLoss += model.loss(x_train[rxy], y_train[rxy])

            y_hat = model.call(x_train[rxy])
            if y_hat.argmax() == y_train[rxy].argmax():
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
    load_model("data/bias_model/best/best.json")

    if True:
        # predict_loop()
        test_confusion()
        train_confusion()
        loss_stats()
    else:
        save_parameters()
        training_loop()
        test_confusion()
        loss_stats()
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from NeuralModel import DoubleModel, BigModel


X = 250


# create linearly seperable dataset
C = [
    [1, -0.78], 
    [-0.78, 1]
]
A = la.cholesky(C)
xa = np.random.randn(X//2, 2)
xa = np.dot(xa, A)

xb = np.random.randn(X//2, 2)
xb = np.dot(xb, A) + 2

x = np.vstack([xa, xb])
x = np.interp(x, (x.min(), x.max()), (0, 1))
y = np.hstack([np.zeros(X//2), np.ones(X//2)]).reshape(-1, 1)
y = np.hstack([y, 1-y])
# y = [
#     [1, 0],
#     ...
#     [0, 1],
#     ...
# ]


def plot_confusion_matrix(y1Data, y2Data):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    confMat = confusion_matrix(y1Data, y2Data)

    ax.matshow(confMat, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(confMat.shape[0]):
        for j in range(confMat.shape[1]):
            plt.text(x=j, y=i, s=confMat[i, j], va='center', ha='center')

    plt.xlabel("True Prediction")
    plt.ylabel("Model Prediction")

    plt.savefig("images/double_separation_confused.png")
    plt.close()


def training_loop(x, y):
    model = BigModel(1)
    model.init_weights()

    # set initial weights
    for rxy in range(len(x)):
        model.grad(x[rxy], y[rxy])

    for e in range(20):

        # initial pass
        correct = np.zeros(y.shape)
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            correct[rxy] = ((y_hat.argmax()) == (y[rxy][1] == 1)) * 1

        # train incorrect points
        for rxy in np.where(correct < 1)[0]:
            grad1, grad2 = model.grad(x[rxy], y[rxy])

        correct = np.zeros(y.shape[0])
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            if (y_hat.argmax()) == (y[rxy][1] == 1):
                correct[rxy] = 1

        print("Epoch {}: {}/{}".format(e, correct.sum(), len(correct)))
    
    # evaluate model
    correct = np.zeros(y.shape[0])
    y_hat = np.zeros(y.shape)
    for rxy in range(len(x)):
        y_hat[rxy] = model.call(x[rxy])

        if (y_hat[rxy].argmax()) == (y[rxy][1] == 1):
            correct[rxy] = 1

    plot_confusion_matrix(y_hat.argmax(axis=1), y[:,1])

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))

    # print(model.layer1.weights)
    # print(model.layer2.weights)


def simple_training_loop(x, y):
    model = BigModel(0.01)
    model.init_weights()

    
    for e in range(20):

        # train points
        for rxy in range(len(x)):
            d1, d2, d3 = model.grad(x[rxy], y[rxy])
            # print("Grad: ", d1, d2, d3)

        correct = np.zeros(y.shape[0])
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            if (y_hat.argmax()) == (y[rxy][1] == 1):
                correct[rxy] = 1

        print("Epoch {}: {}/{}".format(e, correct.sum(), len(correct)))
    
    # evaluate model
    correct = np.zeros(y.shape[0])
    y_hat = np.zeros(y.shape)
    for rxy in range(len(x)):
        y_hat[rxy] = model.call(x[rxy])

        if (y_hat[rxy].argmax()) == (y[rxy][1] == 1):
            correct[rxy] = 1

    plot_confusion_matrix(y_hat.argmax(axis=1), y[:,1])

    print(f"1s: {y_hat.argmax(axis=1).sum()} ")

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))



    print(model.layer1.weights)
    print(model.layer2.weights)
    print(model.layer3.weights)

if __name__ == "__main__":
    # training_loop(x, y)
    simple_training_loop(x, y)
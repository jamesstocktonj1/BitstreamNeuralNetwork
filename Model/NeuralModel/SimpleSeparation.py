import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from NeuralModel import SimpleModel


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

    plt.savefig("images/simple_separation_confused.png")
    plt.close()


def training_loop(x, y):
    model = SimpleModel(0.000125)
    model.init_weights()

    # set initial weights
    for rxy in range(len(x)):
        model.grad(x[rxy], y[rxy])

    for e in range(10):

        # initial pass
        correct = np.zeros(y.shape)
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

        # train incorrect points
        for rxy in np.where(correct < 1)[0]:
            grad1, grad2 = model.grad(x[rxy], y[rxy])

        correct = np.zeros(y.shape)
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

        print("Epoch {}: {}/{}".format(e, correct.sum(), len(correct)))
    
    # evaluate model
    correct = np.zeros(y.shape)
    y_hat = np.zeros(y.shape)
    for rxy in range(len(x)):
        y_hat[rxy] = model.call(x[rxy])

        if (y_hat[rxy] > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    plot_confusion_matrix(y_hat > 0.5, y)

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))



if __name__ == "__main__":
    training_loop(x, y)
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix

from NeuralModel import SimpleModel, HiddenModel


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


def plot_decision(model):
    plt.figure()

    for i in range(50):
        for j in range(50):
            x = np.array([i/50, j/50])
            y = model.call(x)

            if y > 0.5:
                plt.plot(i, j, '-ro')
            else:
                plt.plot(i, j, '-bx')

    plt.savefig("images/simple_decision.png")
    plt.close()

def plot_3d_plane(model):

    fig = plt.figure()
    fig1 = fig.add_subplot(projection='3d')

    x = np.arange(0, 1, 1/X)
    y = np.arange(0, 1, 1/X)
    x_t, y_t = np.meshgrid(x, y)

    # neuronPlane = call_neuron(x_t, y_t, n)
    neuronPlane = np.zeros((X, X))

    for i in range(X):
        for j in range(X):
            neuronPlane[i,j] = model.call(np.array([x_t[i,j], y_t[i,j]]))

    surf = fig1.plot_surface(x_t, y_t, neuronPlane, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # surf = fig1.plot_surface(x_t, y_t, neuronPlane, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=None, antialiased=True)

    fig.colorbar(surf)
    plt.show()

def training_loop(x, y):
    model = HiddenModel(0.0025)

    print(model.layer1.weights)
    print(model.layer2.weights)
    # model.init_weights()

    # set initial weights
    for rxy in range(len(x)):
        model.grad(x[rxy], y[rxy])

    for e in range(20):

        # initial pass
        correct = np.zeros(y.shape)
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            correct[rxy] = ((y_hat[0] > 0.5) == (y[rxy] == 1)) * 1

        # train incorrect points
        for rxy in np.where(correct < 1)[0]:
            grads = model.grad(x[rxy], y[rxy])
            # print(grads)

        correct = np.zeros(y.shape)
        modelLoss = 0
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            if (y_hat[0] > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

            modelLoss += model.loss(x[rxy], y[rxy])[0]

        print("Epoch {}: {}/{}".format(e, correct.sum(), len(correct)))
        print("      {}".format(modelLoss / X))
    
    # evaluate model
    correct = np.zeros(y.shape)
    y_hat = np.zeros(y.shape)
    for rxy in range(len(x)):
        y_hat[rxy] = model.call(x[rxy])

        if (y_hat[rxy][0] > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    plot_confusion_matrix(y_hat > 0.5, y)

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))

    print(model.layer1.weights)
    print(model.layer2.weights)

    plot_decision(model)
    # plot_3d_plane(model)



if __name__ == "__main__":
    training_loop(x, y)
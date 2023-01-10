import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix

from NeuralModel import SimpleModel, HiddenModel, DeepModel, DeepDeepModel


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
# y = np.hstack([np.ones(X//2), np.zeros(X//2)]).reshape(-1, 1)


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

def plot_3d_plane_points(model, x_data, y_data):

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

    fig1.scatter(x_data[X//2:,0], x_data[X//2:,1], np.ones(X//2) * 0.5, 'ro')
    fig1.scatter(x_data[:X//2,0], x_data[:X//2,1], np.ones(X//2) * 0.5, 'bo')

    fig.colorbar(surf)
    plt.show()

def plot_loss_epoch(correct, loss):
    plt.figure()

    fig1 = plt.subplot(2, 1, 1)
    fig2 = plt.subplot(2, 1, 2)

    fig1.plot(np.arange(len(correct)), correct)
    fig2.plot(np.arange(len(loss)), loss)

    plt.savefig("images/simple_epochs.png")
    plt.close()

def plot_grad_epoch(grad):
    plt.figure()

    grad1 = list(g[0] for g in grad)
    grad2 = list(g[1] for g in grad)
    grad3 = list(g[2] for g in grad)

    plt.plot(np.arange(len(grad1)), grad1)
    plt.plot(np.arange(len(grad2)), grad2)
    plt.plot(np.arange(len(grad3)), grad3)

    plt.savefig("images/simple_grad.png")
    plt.close()

def training_loop(x, y):
    # plot_grad_epoch(gradEpoch)
    # model = HiddenModel(0.025)
    # model = DeepModel(0.0125)
    model = DeepDeepModel(0.00125)

    print(model.layer1.weights)
    print(model.layer2.weights)
    # model.init_weights()

    # set initial weights
    for rxy in range(len(x)):
        model.grad(x[rxy], y[rxy])

    correctEpoch = []
    lossEpoch = []
    for e in range(250):

        # initial pass
        correct = np.zeros(y.shape)
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            correct[rxy] = ((y_hat[0] > 0.5) == (y[rxy] == 1)) * 1

        # train incorrect points
        for rxy in np.where(correct < 1)[0]:
            grads = model.grad(x[rxy], y[rxy])
            # print(grads)

        # tempGrad = 0
        # for rxy in range(len(x)):
        #     grads = model.grad(x[rxy], y[rxy])
            # tempGrad += grads[1][0]
        # gradEpoch.append(tempGrad / len(x))
        

        correct = np.zeros(y.shape)
        modelLoss = 0
        for rxy in range(len(x)):
            y_hat = model.call(x[rxy])

            if (y_hat[0] > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

            modelLoss += model.loss(x[rxy], y[rxy])[0]
        
        correctEpoch.append(correct.sum())
        lossEpoch.append(modelLoss / X)

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
    print(model.layer3.weights)
    
    plot_loss_epoch(correctEpoch, lossEpoch)

    plot_decision(model)
    # plot_grad_epoch(gradEpoch)
    # plot_3d_plane(model)
    # plot_3d_plane_points(model, x, y)



if __name__ == "__main__":
    training_loop(x, y)
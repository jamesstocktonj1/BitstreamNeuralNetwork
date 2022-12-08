import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm

from Model import *

# parameters
N = list(2**n for n in range(4, 12))
X = 250
trainingRate = 0.000075


# create normal dataset
# x = np.vstack([np.random.randn(X//2, 2)/6 + 0.25, np.random.randn(X//2, 2)/6 + 0.75])
# x = np.clip(x, 0, 1.0)
# y = np.hstack([-np.ones(X//2), np.ones(X//2)]).reshape(-1, 1)

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
y = np.hstack([-np.ones(X//2), np.ones(X//2)]).reshape(-1, 1)


# plot points
def plot_data(x, y, N):
    plt.figure()
    plt.plot(x[:X//2, 0], x[:X//2, 1], 'ro')
    plt.plot(x[X//2:, 0], x[X//2:, 1], 'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_training_points_{}bits.png".format(N))
    plt.close()


def train_like_neuron_loop(x, y, r, epochs):

    model = HiddenPerceptron()

    weights = np.ones((6, )) * 0.5
    model.set_weights(weights)

    for e in range(epochs):
        correct = np.zeros(y.shape)
        i = 0

        model.set_weights(weights)

        # initial correct points
        for rxy in range(len(x)):
            y_hat = model.call(np.array([x[rxy][0], x[rxy][1]]))

            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

        print("Pass: {}\tIncorrect Points: {}".format(e, np.sum(correct == 0)))

        # train incorrect points
        while np.sum(correct) < len(y):
            rxy = np.random.choice(np.where(correct < 1)[0])
            i += 1

            grad = model.back_prop(x[rxy], y[rxy])

            weights -= r * grad
            weights = np.clip(weights, 0, 1.0)
            model.set_weights(weights)

            # print("Weights: ", weights)

            # run through neuron
            y_hat = model.call(x[rxy])

            # if correctly classified
            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

            # leave early
            if i > (X * 10):
                break

            print("{}/{}".format(correct.sum(), X))

    # evaluate model
    correct = np.zeros(y.shape)
    model.set_weights(weights)     # re-randomise the weights
    for rxy in range(len(x)):
        y_hat = model.call(np.array([x[rxy][0], x[rxy][1]]))

        if (y_hat > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))

    return weights, model


def perceptron_flood(neuron):
    plt.figure()

    x = np.arange(0, 1, 1/X)
    y = np.arange(0, 1, 1/X)
    x, y = np.meshgrid(x, y)

    for i in range(X):
        for j in range(X):
            z = neuron.call(np.array([x[i,j], y[i,j]]))

            if z > 0.5:
                plt.plot(i, j, 'ro')
            else:
                plt.plot(i, j, 'bo') 

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_flood_{}bits.png".format(N))
    plt.close()

def plot_decision(neuron, x, y):
    plt.figure()

    x_t = np.arange(0, 1, 1/X)
    y_t = neuron.decision(x_t)

    plt.plot(x[:X//2, 0], x[:X//2, 1], 'ro')
    plt.plot(x[X//2:, 0], x[X//2:, 1], 'bo')

    plt.plot(x_t, y_t)

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_decision.png")
    plt.close()

def perceptron_dual_plot(x, y, R, N, neuron, weights):
    plt.figure()
    fig1 = plt.subplot(1, 2, 1)
    fig1.set_aspect("equal", adjustable="box")
    fig1.set_xlim([0, 1])
    fig1.set_ylim([0, 1])
    
    fig2 = plt.subplot(1, 2, 2)
    fig2.set_aspect("equal", adjustable="box")
    fig2.set_xlim([0, 1])
    fig2.set_ylim([0, 1])

    # plot points
    fig1.plot(x[:X//2, 0], x[:X//2, 1], 'bo')
    fig1.plot(x[X//2:, 0], x[X//2:, 1], 'ro')

    # plot flood
    x = np.arange(0, 1, 1/X)
    y = np.arange(0, 1, 1/X)
    x, y = np.meshgrid(x, y)

    for i in range(X):
        for j in range(X):
            z = neuron.call(np.array([x[i,j], y[i,j]]))

            if y > 0.5:
                plt.plot(i, j, 'ro')
            else:
                plt.plot(i, j, 'bo') 
    
    fig1.set_title("Weights: {:.4f}, {:.4f}".format(weights[0], weights[1]))
    plt.suptitle("Trained Perceptron {}-Bits".format(N))
    plt.savefig("images/perceptron_linear_data_{}bits.png".format(N))
    plt.close()


def plot_model(model):
    fig = plt.figure()
    fig1 = fig.add_subplot(projection='3d')

    x = np.arange(0, 1, 0.02)
    y = np.arange(0, 1, 0.02)
    x_t, y_t = np.meshgrid(x, y)

    modelPlane = np.zeros((50, 50))

    for i in range(50):
        for j in range(50):
            temp = model.call(np.array([x_t[i,j], y_t[i,j]]))
            modelPlane[i,j] = (temp > 0.5) * 1

    surf = fig1.plot_surface(x_t, y_t, modelPlane, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf)
    plt.show()



def main():
    print("\n\"Real\" Perceptron")
    w, model = train_like_neuron_loop(x, y, trainingRate, 2)
    # perceptron_flood(neuron)
    # plot_decision(neuron, x, y)
    plot_model(model)
    print("Weights: {}".format(w))


if __name__ == "__main__":
    main()
    plot_data(x, y, "Data")
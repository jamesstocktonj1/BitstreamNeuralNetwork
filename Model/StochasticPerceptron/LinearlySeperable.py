import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from Neuron import *

# parameters
N = list(2**n for n in range(4, 12))
X = 500
trainingRate = 0.002


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


def training_loop(x, y, r, N, epochs):
    # setup neuron with initial weights
    neuron = Neuron(2, N, N)
    #w = np.random.randint(0, 20, size=(2)) * 0.05
    weights = np.array([0.5, 0.5])
    neuron.set_weights(weights)

    for e in range(epochs):
        correct = np.zeros(y.shape)
        i = 0

        neuron.set_weights(weights)

        # initial correct points
        for rxy in range(len(x)):
            x_hat0 = bitstream_generator_exact(x[rxy][0], N)
            x_hat1 = bitstream_generator_exact(x[rxy][1], N)
            y_hat = neuron.call(np.array([x_hat0, x_hat1]))
            y_hat = bitstream_integrator(y_hat)

            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

        print("Pass: {}\tIncorrect Points: {}".format(e, np.sum(correct == 0)))

        # train incorrect points
        while np.sum(correct) < len(y):
            rxy = np.random.choice(np.where(correct < 1)[0])
            i += 1

            weights = weights + r * (y[rxy] * x[rxy])
            weights = np.clip(weights, 0, 1.0)
            neuron.set_weights(weights)

            #print(weights)

            # run through neuron
            x_hat0 = bitstream_generator_exact(x[rxy][0], N)
            x_hat1 = bitstream_generator_exact(x[rxy][1], N)
            y_hat = neuron.call(np.array([x_hat0, x_hat1]))
            y_hat = bitstream_integrator(y_hat)

            
            # if correctly classified
            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

            # leave early
            if i > (X * 10):
                break

    # evaluate model
    correct = np.zeros(y.shape)
    neuron.set_weights(weights)     # re-randomise the weights
    for rxy in range(len(x)):
        x_hat0 = bitstream_generator_exact(x[rxy][0], N)
        x_hat1 = bitstream_generator_exact(x[rxy][1], N)
        y_hat = neuron.call(np.array([x_hat0, x_hat1]))
        y_hat = bitstream_integrator(y_hat)

        if (y_hat > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))

    return weights, neuron

def train_like_neuron_loop(x, y, r, epochs):
    # setup neuron with initial weights
    neuron = BitLikeNeuron(2)
    #w = np.random.randint(0, 20, size=(2)) * 0.05
    weights = np.array([0.5, 0.5])
    neuron.set_weights(weights)

    for e in range(epochs):
        correct = np.zeros(y.shape)
        i = 0

        neuron.set_weights(weights)

        # initial correct points
        for rxy in range(len(x)):
            y_hat = neuron.call(np.array([x[rxy][0], x[rxy][1]]))

            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

        print("Pass: {}\tIncorrect Points: {}".format(e, np.sum(correct == 0)))

        # train incorrect points
        while np.sum(correct) < len(y):
            rxy = np.random.choice(np.where(correct < 1)[0])
            i += 1

            weights = weights + r * (y[rxy] * x[rxy])
            weights = np.clip(weights, 0, 1.0)
            neuron.set_weights(weights)

            # run through neuron
            y_hat = neuron.call(np.array([x[rxy][0], x[rxy][1]]))

            # if correctly classified
            if (y_hat > 0.5) == (y[rxy] == 1):
                correct[rxy] = 1

            # leave early
            if i > (X * 10):
                break

    # evaluate model
    correct = np.zeros(y.shape)
    neuron.set_weights(weights)     # re-randomise the weights
    for rxy in range(len(x)):
        y_hat = neuron.call(np.array([x[rxy][0], x[rxy][1]]))

        if (y_hat > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))

    return weights, neuron


def perceptron_flood(R, N, neuron):
    plt.figure()

    for i in range(R):
        i = i / R
        x_0 = bitstream_generator_exact(i, N)

        for j in range(R):
            j = j / R
            x_1 = bitstream_generator_exact(j, N)

            y = neuron.call(np.array([x_0, x_1]))
            y = bitstream_integrator(y)

            if y > 0.5:
                plt.plot(i, j, 'ro')
            else:
                plt.plot(i, j, 'bo')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_flood_{}bits.png".format(N))
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
    for i in range(R):
        i = i / R
        x_0 = bitstream_generator_exact(i, N)

        for j in range(R):
            j = j / R
            x_1 = bitstream_generator_exact(j, N)

            y = neuron.call(np.array([x_0, x_1]))
            y = bitstream_integrator(y)

            if y > 0.5:
                fig2.plot(i, j, 'ro')
            else:
                fig2.plot(i, j, 'bo')
    
    fig1.set_title("Weights: {:.4f}, {:.4f}".format(weights[0], weights[1]))
    plt.suptitle("Trained Perceptron {}-Bits".format(N))
    plt.savefig("images/perceptron_linear_data_{}bits.png".format(N))
    plt.close()


def main():
    for n in N:

        print("\nPerceptron {}-Bits".format(n))
        w, neuron = training_loop(x, y, trainingRate * 0.5, n, 10)
        print("Weights: {}".format(w))

        perceptron_dual_plot(x, y, 50, n, neuron, w)

    print("\n\"Real\" Perceptron")
    w, neuron = train_like_neuron_loop(x, y, trainingRate * 0.5, 10)
    print("Weights: {}".format(w))


if __name__ == "__main__":
    main()
    plot_data(x, y, "Data")

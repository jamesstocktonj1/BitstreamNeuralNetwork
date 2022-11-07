import numpy as np
import matplotlib.pyplot as plt

from Neuron import *

# parameters
N = list(2**n for n in range(4, 12))
X = 100
trainingRate = 0.1


# create dataset
x = np.vstack([np.random.randint(0, 20, size=(X//2, 2))*0.025, np.random.randint(20, 40, size=(X//2, 2))*0.025])
y = np.hstack([np.zeros(X//2), np.ones(X//2)]).reshape(-1, 1)


# plot points
def plot_data(x, y, N):
    plt.figure()
    plt.plot(x[:X//2, 0], x[:X//2, 1], 'ro')
    plt.plot(x[X//2:, 0], x[X//2:, 1], 'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_training_points_{}bits.png".format(N))
    plt.close()


def training_loop(x, y, r, N):
    # setup neuron with initial weights
    neuron = Neuron(2, N, N)
    w = np.array([0.5, 0.5])
    neuron.set_weights(w)

    correct = -np.ones(y.shape)
    i = 0

    # initial correct points
    for rxy in range(len(x)):
        x_hat0 = bitstream_generator_exact(x[rxy][0], N)
        x_hat1 = bitstream_generator_exact(x[rxy][1], N)
        y_hat = neuron.call(np.array([x_hat0, x_hat1]))
        y_hat = bitstream_integrator(y_hat)

        if (y_hat > 0.5) == y[rxy]:
            correct[rxy] = 1

    weights = np.array([0.1, 0.1])

    # train incorrect points
    while np.sum(correct) < len(y):
        rxy = np.random.choice(np.where(correct < 1)[0])
        i += 1

        weights = weights + r * (y[rxy] * x[rxy])
        weights = np.clip(weights, 0, 1.0)
        neuron.set_weights(weights)

        print(weights)
        

        # run through neuron
        x_hat0 = bitstream_generator_exact(x[rxy][0], N)
        x_hat1 = bitstream_generator_exact(x[rxy][1], N)
        y_hat = neuron.call(np.array([x_hat0, x_hat1]))
        y_hat = bitstream_integrator(y_hat)

        
        # if correctly classified
        if (y_hat > 0.5) == y[rxy]:
            correct[rxy] = 1

        # leave early
        if i > X:
            break

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




for n in N:

    w, neuron = training_loop(x, y, trainingRate, n)
    print("\nPerceptron {}-Bits".format(n))
    print("Weights: {}".format(w))

    plot_data(x, y, n)
    perceptron_flood(50, n, neuron)

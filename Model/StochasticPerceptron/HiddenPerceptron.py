import numpy as np
import matplotlib.pyplot as plt

from Neuron import *

# parameters
N = list(2**n for n in range(4, 12))
X = 1000
trainingRate = 0.01


# create normal dataset
x = np.vstack([np.random.randn(X//2, 2)/6 + 0.25, np.random.randn(X//2, 2)/6 + 0.75])
x = np.clip(x, 0, 1.0)
y = np.hstack([np.zeros(X//2), np.ones(X//2)]).reshape(-1, 1)


# plot points
def plot_data(x, y, N):
    plt.figure()
    plt.plot(x[:X//2, 0], x[:X//2, 1], 'ro')
    plt.plot(x[X//2:, 0], x[X//2:, 1], 'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_hidden_points_{}bits.png".format(N))
    plt.close()



class PerceptronModel:

    def __init__(self, N):
        self.dense1_1 = Neuron(2, N, N)
        self.dense1_2 = Neuron(2, N, N)

        self.dense2 = Neuron(2, N, N)

    def set_weights(self):
        self.dense1_1.set_weights(np.array([1.0000000, 0.32587668]))
        self.dense1_2.set_weights(np.array([0.5567364, 0.56139076]))
        self.dense2.set_weights(np.array([0.38991064, 0.56984866]))

    def call(self, x):
        a = self.dense1_1.call(x)
        b = self.dense1_2.call(x)

        x = self.dense2.call(np.array([a, b]))

        return x

class RealModel:

    def __init__(self):
        self.dense1_1 = BitLikeNeuron(2)
        self.dense1_2 = BitLikeNeuron(2)

        self.dense2 = BitLikeNeuron(2)

    def set_weights(self):
        self.dense1_1.set_weights(np.array([1.0000000, 0.32587668]))
        self.dense1_2.set_weights(np.array([0.5567364, 0.56139076]))
        self.dense2.set_weights(np.array([0.38991064, 0.56984866]))

    def call(self, x):
        a = self.dense1_1.call(x)
        b = self.dense1_2.call(x)

        x = self.dense2.call(np.array([a, b]))

        return x




def training_loop(x, y, N):


    model = PerceptronModel(N)

    # evaluate model
    correct = np.zeros(y.shape)
    model.set_weights()
    for rxy in range(len(x)):
        x_hat0 = bitstream_generator_exact(x[rxy][0], N)
        x_hat1 = bitstream_generator_exact(x[rxy][1], N)
        y_hat = model.call(np.array([x_hat0, x_hat1]))
        y_hat = bitstream_integrator(y_hat)

        if (y_hat > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))

    return model


def perceptron_flood(R, N, model):
    plt.figure()

    for i in range(R):
        i = i / R
        x_0 = bitstream_generator_exact(i, N)

        for j in range(R):
            j = j / R
            x_1 = bitstream_generator_exact(j, N)

            y = model.call(np.array([x_0, x_1]))
            y = bitstream_integrator(y)

            if y > 0.5:
                plt.plot(i, j, 'ro')
            else:
                plt.plot(i, j, 'bo')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_flood_hidden_{}bits.png".format(N))
    plt.close()

def real_perceptron_flood(R, N, model):
    plt.figure()

    for i in range(R):
        x_0 = i / R

        for j in range(R):
            x_1 = j / R

            y = model.call(np.array([x_0, x_1]))

            if y > 0.5:
                plt.plot(i, j, 'ro')
            else:
                plt.plot(i, j, 'bo')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("images/perceptron_flood_hidden_{}bits.png".format(N))
    plt.close()


def perceptron_dual_plot(x, y, R, N, neuron, weights):
    plt.figure()
    fig1 = plt.subplot(1, 2, 1)
    fig1.set_aspect("equal", adjustable="box")
    
    fig2 = plt.subplot(1, 2, 2)
    fig2.set_aspect("equal", adjustable="box")

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
    plt.savefig("images/perceptron_dual_{}bits.png".format(N))
    plt.close()


def main():
    for n in N:

        print("\nPerceptron {}-Bits".format(n))
        model = training_loop(x, y, n)
        #print("Weights: {}".format(w))

        perceptron_flood(50, n, model)
        plot_data(x, y, n)


        #perceptron_dual_plot(x, y, 50, n, neuron, w)

    print("\nReal Perceptron")
    model = RealModel()
    model.set_weights()

    real_perceptron_flood(50, "real", model)

if __name__ == "__main__":
    main()

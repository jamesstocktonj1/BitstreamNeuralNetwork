import numpy as np
import matplotlib.pyplot as plt

from Neuron import *


N = list(2**n for n in range(4, 11))
M = list(2**n for n in range(1, 11))
X = 50


# iterate through bitstream length
for n in N:

    plt.figure()
    plt.subplots_adjust(hspace=0.4)

    fig1 = plt.subplot(2, 1, 1)
    fig1.set_title("Variance")
    fig1.set_aspect("auto", adjustable="box")

    fig2 = plt.subplot(2, 1, 2)
    fig2.set_title("Standard Deviation")
    fig2.set_aspect("auto", adjustable="box")

    var = []
    sd = []

    # iterate through input count
    for m in M:

        temp_var = 0
        dif = 0

        # iterate through test count
        for i in range(X):

            # m number of random numbers
            a = []
            b = np.random.rand(m)

            # create bitstreams
            for j in b:
                a.append(bitstream_generator_exact(j, n))

            a = np.array(a)
            y = bitstream_integrator(bitstream_mux_sum(a))
            y_n = b.sum() / m

            temp_var += (y - y_n) ** 2
            dif += abs(y - y_n)

        temp_var = temp_var / X

        var.append(temp_var)
        sd.append(temp_var ** 0.5)
        print("Bitstream: {}\tInput Count: {}\tDifference: {}".format(n, m, dif / X))

    fig1.plot(M, var)
    fig2.plot(M, sd)

    plt.savefig("images/variance_mux_{}bit.png".format(n))
    plt.close()

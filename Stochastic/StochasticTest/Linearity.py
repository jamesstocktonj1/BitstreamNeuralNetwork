import numpy as np
import matplotlib.pyplot as plt

from Neuron import *


N = list(2**n for n in range(4, 11))
X = 50


gradients = [0.1, 0.3, 0.6]
line_colors = ['c', 'm', 'y']


for n in N:
    plt.figure()

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # iterate through gradients
    for m, cl in zip(gradients, line_colors):

        x = list(i/X for i in range(X))
        y = []
        y1 = []
        
        b = bitstream_generator_exact(m, n)
        
        for i in range(X):
            i = i / X

            a = bitstream_generator_exact(i, n)
            
            #c = (a == 1) & (b == 1)
            #c = (a == 1) | (b == 1)
            # c = bitstream_mux_or(a, b)
            c = (a == 1) != (b == 1)

            c = bitstream_integrator(c)
            y.append(c)

            #y1.append(i * m)
            #y1.append(i + m - (i * m))
            # y1.append((i + m) / 2)
            temp = (2 * i) - 1
            temp *= c
            temp = (temp * 0.5) + 0.5
            y1.append(temp)

        plt.plot(x, y, color=cl)
        plt.plot(x, y1, color=cl, linestyle='dashed')

    plt.savefig("images/linearity_sum_xor_{}bit.png".format(n))
    plt.close()

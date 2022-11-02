import numpy as np
import matplotlib.pyplot as plt

from Neuron import *


N = 512

X = 50


gradients = [0.1, 0.3, 0.6]
line_colors = ['c', 'm', 'y']

plt.ylim(0, 1)

# iterate through gradients
for m, cl in zip(gradients, line_colors):

    x = list(range(X))
    y = []
    y1 = []
    
    b = bitstream_generator_exact(m, N)
    
    for i in range(X):
        i = i / X

        a = bitstream_generator_exact(i, N)
        
        c = (a == 1) | (b == 1)
        c = bitstream_integrator(c)

        y.append(c)
        y1.append(i + m)

    plt.plot(x, y, color=cl)
    plt.plot(x, y1, color=cl, linestyle='dashed')

plt.savefig("images/linearity_sum_{}bit.png".format(N))
#plt.show()

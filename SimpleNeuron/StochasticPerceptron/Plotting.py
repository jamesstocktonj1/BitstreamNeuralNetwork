import numpy as np
import matplotlib.pyplot as plt
from Neuron import *

N = 64


w = [0.35, 0.94]
z = np.zeros(shape=(2, N), dtype=np.int32)

neuron = Neuron(2, N, N)
neuron.set_weights(w)


X = 25
for x in range(X):
    for y in range(X):

        z[0] = bitstream_generator(x/X, N)
        z[1] = bitstream_generator(y/X, N)

        a = []
        for i in range(20):
            w = neuron.call(z)
            w = bitstream_integrator(w)

            a.append(w)

        o = sum(a) / len(a)

        print("X: {}\tY:{}\tZ: {}".format(x/X, y/X, o))
        if o > 0.5:
            plt.plot(x, y, 'rx') 
        else:
            plt.plot(x, y, 'bo')

plt.show()
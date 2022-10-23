import numpy as np
import matplotlib.pyplot as plt
from Neuron import *

N = 64


w = [0.35, 0.94]
z = np.zeros(shape=(2, N), dtype=np.int32)

neuron = Neuron(2, N, N)
neuron.set_weights(w)

realNeuron = RealNeuron(np.array(w))


fig1 = plt.subplot(1, 2, 1)
fig1.set_title("Bit-Neuron")
fig1.set_aspect("equal", adjustable="box")

fig2 = plt.subplot(1, 2, 2)
fig2.set_title("Real-Neuron")
fig2.set_aspect("equal", adjustable="box")

X = 25
for x in range(X):
    for y in range(X):

        z[0] = bitstream_generator(x/X, N)
        z[1] = bitstream_generator(y/X, N)

        a = []
        for i in range(1):
            w = neuron.call(z)
            w = bitstream_integrator(w)

            a.append(w)

        o = sum(a) / len(a)
        if o > 0.5:
            fig1.plot(x, y, 'rx') 
        else:
            fig1.plot(x, y, 'bo')

        o = realNeuron.call([x/X, y/X])
        if o > 0.5:
            fig2.plot(x, y, 'rx') 
        else:
            fig2.plot(x, y, 'bo')
            
plt.suptitle("{}-Bit Perceptron Comparison".format(N))
plt.savefig("perceptron_comparison.png")
#plt.show()
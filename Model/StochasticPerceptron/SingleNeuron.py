



import numpy as np
import matplotlib.pyplot as plt
from Neuron import *

N = 16


w = [0.35]
#w = [0.55, 0.35]
z = np.zeros(shape=(1, N), dtype=np.int32)

neuron = Neuron(1, N, N)
neuron.set_weights(w)

realNeuron = RealNeuron(np.array(w))


fig1 = plt.subplot(1, 2, 1)
fig1.set_title("Bit-Neuron")
fig1.set_aspect("equal", adjustable="box")

fig2 = plt.subplot(1, 2, 2)
fig2.set_title("Real-Neuron")
fig2.set_aspect("equal", adjustable="box")

X = 50

for x in range(X):

    z[0] = bitstream_generator_exact(x/X, N)

    o = neuron.call(z)[0]
    if o > 0.5:
        fig1.plot(x, 0, 'ro') 
    else:
        fig1.plot(x, 0, 'bo')

    o = realNeuron.call([x/X])
    if o > 0.5:
        fig2.plot(x, 0, 'ro') 
    else:
        fig2.plot(x, 0, 'bo')
            
plt.suptitle("{}-Bit Perceptron Comparison".format(N))
plt.savefig("images/perceptron_single_{}bit_exact.png".format(N))
#plt.show()
import numpy as np
import matplotlib.pyplot as plt
from Neuron import *

N = 128
X = 50


w = [0.35, 0.94]
#w = [0.55, 0.35]
z = np.zeros(shape=(2, N), dtype=np.int32)

neuron = Neuron(2, N, N)
neuron.set_weights(w)

realNeuron = RealNeuron(np.array(w))


fig1 = plt.subplot(1, 2, 1)
fig1.set_title("Bit-Neuron")
fig1.set_aspect("equal", adjustable="box")
fig1.set_xlim([0, X])
fig1.set_ylim([0, X])

fig2 = plt.subplot(1, 2, 2)
fig2.set_title("False Classification")
fig2.set_aspect("equal", adjustable="box")
fig2.set_xlim([0, X])
fig2.set_ylim([0, X])

falseCount = 0
for x in range(X):
    for y in range(X):

        z[0] = bitstream_generator(x/X, N)
        z[1] = bitstream_generator(y/X, N)

        a = []
        for i in range(1):
            w = neuron.call(z)
            w = bitstream_integrator(w)

            a.append(w)

        j = sum(a) / len(a)
        k = realNeuron.call([x/X, y/X])
        if j > 0.5:
            fig1.plot(x, y, 'go') 
        else:
            fig1.plot(x, y, 'bo')

        
        if (k > 0.5) != (j > 0.5):
            falseCount += 1
            fig2.plot(x, y, 'ro')


print("Bitstream Length: {}".format(N))
print("Range Count: {}".format(X ** 2))
print("False Classifications: {}".format(falseCount))
print("False Classifications: {}%".format((falseCount * 100) / (X ** 2)))

plt.suptitle("{}-Bit Perceptron Comparison".format(N))
plt.savefig("images/perceptron_false_{}bit.png".format(N))
#plt.show()
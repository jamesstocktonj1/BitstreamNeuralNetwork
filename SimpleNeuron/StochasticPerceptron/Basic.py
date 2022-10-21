import numpy as np
import matplotlib.pyplot as plt
from Neuron import *

N = 8

x = np.array([
    [1,0,0,1,0,1,1,0],
    [1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0]
])


W = np.array([
    [1,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0]
])



w = [0.25, 0.5, 0.6, 0.7]
X = np.array([1, 0, 0, 0])
z = np.zeros(shape=(4, N), dtype=np.int32)

for i in range(len(X)):
    z[i] = np.array(bitstream_generator(X[i], N))


z = np.array(z)

neuron = Neuron(4, 8, 8)
#neuron.set_weights(w)

neuron.weights = W

y = neuron.call(z)

print(y)
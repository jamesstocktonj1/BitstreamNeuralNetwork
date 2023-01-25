from operator import xor
import numpy as np
import matplotlib.pyplot as plt

from Neuron import *


N = list(2**n for n in range(4, 11))
X = 50

for n in N:
    # calculation results
    z = []
    var = 0
    
    a = np.zeros(n, dtype=np.int32)
    b = np.zeros(n, dtype=np.int32)
    x = np.zeros(n, dtype=np.int32)

    # itterate through x y
    for i in range(1, X):
        i = i / X
        for j in range(1, X):
            j = j / X

            a = bitstream_generator_exact(i, n)
            b = bitstream_generator_exact(j, n)

            # c = bitstream_mux_or(a, b)
            c = (a == 1) != (b == 1)

            #c = (a == 1) & (b == 1)
            #c = (a == 1) | (b == 1)
            c = bitstream_integrator(c)

            y = i * j

            var += (c - y) ** 2

            z.append(((c - y) / y) * 100)
            
    z = np.array(z)


    var = (var / z.size)

    thresholds = [5, 10]



    print("\nStochastic Multiplication Test")
    print("Bit Depth: {}".format(n))
    print("Data Points: {}".format(z.size))
    print("Variance: {}".format(var))
    print("Standard Deviation: {}".format(var ** 0.5))

    for t in thresholds:
        greater_than = z[z > t]
        less_than = z[z < (-1 * t)]

        print("Data Within {}%: {}".format(t, ((z.size - (greater_than.size + less_than.size)) / z.size) * 100))
        print("Data Above {}%: {}".format(t, (greater_than.size / z.size) * 100))
        print("Data Below {}%: {}".format(t, (less_than.size / z.size) * 100))

from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Neuron import *

N = list(2**n for n in range(4, 12))
X = 50
W = [0.45, 0.35]



def call_neuron(x, y, n):
    neuron = Neuron(2, n, n)
    neuron.set_weights(W)

    neuronPlane = np.zeros((X, X))
    z = np.zeros((2,n))

    for i in range(X):
        for j in range(X):
        
            z[0] = bitstream_generator_exact(x[i,j], n)
            z[1] = bitstream_generator_exact(y[i,j], n)

            y_t = neuron.call(z)
            neuronPlane[i,j] = bitstream_integrator(y_t)

    return neuronPlane

def call_real(x, y):
    
    return (x * W[0]) + (y * W[1]) - (x * W[0] * y * W[1])


def plot_3d_plane(n):

    fig = plt.figure()
    fig1 = fig.add_subplot(projection='3d')

    x = np.arange(0, 1, 1/X)
    y = np.arange(0, 1, 1/X)
    x_t, y_t = np.meshgrid(x, y)

    # neuronPlane = call_neuron(x_t, y_t, n)
    neuronPlane = (call_real(x_t, y_t) > 0.5) * 1

    surf = fig1.plot_surface(x_t, y_t, neuronPlane, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # surf = fig1.plot_surface(x_t, y_t, neuronPlane, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=None, antialiased=True)

    fig.colorbar(surf)
    plt.show()



if __name__ == "__main__":
    plot_3d_plane(10)

    # for n in N:
    #     plot_3d_plane(n)
            

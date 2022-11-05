import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from layers.BitInput import BitInput
from layers.BitOutput import BitOutput
from layers.BitLayer import BitLayer


# constants
N = 256         # number of bits
X = 50         # plot range


class SingleNeuron(Model):

    def __init__(self, N):
        super(SingleNeuron, self).__init__()

        self.bit_creator = BitInput(N)
        self.bit_dense = BitLayer(3, N)
        self.bit_destroyer = BitOutput()

    def call(self, x):
        
        x = self.bit_creator(x)
        x = self.bit_dense(x)
        x = self.bit_destroyer(x)

        return x


model = SingleNeuron(N)
model(tf.constant([0, 0]))
model.bit_dense.kernel = tf.Variable([
    [0.35, 0.94],
    [0.45, 0.84],
    [0.55, 0.74]
])


for i in range(X):
    for j in range(X):
        x = tf.constant([i/X, j/X])
        y = model(x)

        if y[0] > 0.5:
            plt.plot(x[0], x[1], 'ro')
        else:
            plt.plot(x[0], x[1], 'bo')

plt.savefig("images/tf_bitlayer_{}.png".format(N))
#plt.show()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

from layers.BitInput import *
from layers.BitLayer import *
from layers.BitOutput import *
from layers.Bitstream import *


# parameters
N = list(2**n for n in range(4, 12))
X = 100


# create normal dataset
x = np.vstack([np.random.randn(X//2, 2)/6 + 0.25, np.random.randn(X//2, 2)/6 + 0.75])
x = np.clip(x, 0, 1.0)
y = np.hstack([-np.ones(X//2), np.ones(X//2)]).reshape(-1, 1)


class PerceptronModel(Model):

    def __init__(self, N):
        super(PerceptronModel, self).__init__()

        self.input_layer = BitInput(N)
        self.dense = BitLayer(1, N)
        self.output_layer = BitOutput()

        #self.dense = Bitstream(1, N)

    def call(self, x):
        print("Input: ", x)
        x = self.input_layer(x)
        print("Bitstream: ", x)
        x = self.dense(x)
        print("Post Dense: ", x)
        x = self.output_layer(x)
        
        return x

def train_model(x, y, N):

    model = PerceptronModel(N)

    model.build((2, ))
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=3)

    model.summary()
    



def main():

    train_model(x, y, 128)
    '''

    for n in N:

        print("\nPerceptron {}-Bits".format(n))
        w, neuron = training_loop(x, y, trainingRate, n, 5)
        print("Weights: {}".format(w))

        perceptron_dual_plot(x, y, 50, n, neuron, w)
    '''

if __name__ == "__main__":
    main()

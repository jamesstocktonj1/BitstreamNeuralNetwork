import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from BitLayer import BitNeurons




class BitModel(Model):

    def __init__(self):
        super(BitModel, self).__init__()

        self.input_layer = BitNeurons(4)
        self.dense1 = layers.Dense(4)

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense1(x)

        return x


model = BitModel()

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.build(input_shape=(2, 2))

#i = np.array([[1.0], [0.0]])
#print(i)
#model(i)

x_in = layers.Input(shape=(2, ))
model(x_in)

model.summary()

print(model.input_layer.weights)
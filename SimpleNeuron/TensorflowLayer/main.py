import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from BitLayer import BitNeurons




class BitModel(Model):

    def __init__(self):
        super(BitModel, self).__init__()

        self.input_layer = BitNeurons(1)
        # self.dense1 = layers.Dense(1, activation=tf.nn.softmax)

    def call(self, x):
        x = self.input_layer(x)
        # x = self.dense1(x)

        return x

# create model
model = BitModel()

# set input size
x_in = layers.Input(shape=(2, ))
model(x_in)


# manually set weights
weights = np.array([[0.2], [0.3]])
model.input_layer.set_weights([weights])

print(model.input_layer.get_weights())


# test data
inputs = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]

for i in inputs:
    dat = tf.convert_to_tensor([i])
    res = model(dat)
    
    print("\nTest {}".format(i))
    print("Input ", dat.numpy()[0])
    print("Output ", res.numpy()[0])


# print model summary
model.summary()

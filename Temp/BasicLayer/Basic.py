import numpy as np
import tensorflow as tf
from Model import BitModel
from Layer import BitLayer


x = tf.ones((2, 4), dtype=tf.int8)


model = BitModel()
y = model(x)

model.dense1.kernel = tf.Variable([
    [[1,0,0,0],[0,0,1,0]],
    [[1,0,1,0],[0,1,0,0]],
    [[1,0,0,1],[0,1,1,0]],
    [[0,1,0,1],[0,1,0,0]]
])

y = model(x)
print(x)
print(y)

model.summary()
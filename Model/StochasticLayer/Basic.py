import tensorflow as tf
import numpy as np

from layers.BitInput import BitInput, bitinput_test
from layers.BitOutput import BitOutput

x = tf.constant([
    [0.5, 0.3, 0.2]
])
layer1 = BitInput(128)
layer2 = BitOutput()

print(x)
x = layer1(x)
print(x)
y = layer2(x)
print(y)
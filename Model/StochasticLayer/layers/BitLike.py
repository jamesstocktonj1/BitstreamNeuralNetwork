import tensorflow as tf
from tensorflow.keras.layers import Layer



class BitLayer(Layer):

    def __init__(self, num_outputs):
        super(BitLayer, self).__init__()

        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(input_shape[-1], self.num_outputs))

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
        #return 1 - tf.math.reduce_prod(1 - (self.kernel * inputs))


import tensorflow as tf
from tensorflow.keras.layers import Layer


def bitstream_generator_exact(p, N):
    n_bits = int(tf.round(p * N))
    bs = tf.concat([
        tf.ones((n_bits)),
        tf.zeros((N - n_bits))
    ], 0)
    return tf.random.shuffle(bs)


class Bitstream(Layer):

    def __init__(self, num_outputs, bit_size):
        super(Bitstream, self).__init__()

        self.num_outputs = num_outputs
        self.bit_size = bit_size

    def build(self, input_shape):
        # input_shape[-1] is bit_size
        # input_shape[-2] is number of input streams
        self.kernel = self.add_weight(shape=(self.num_outputs, input_shape[-1]), trainable=True, initializer=tf.keras.initializers.Zeros())

    def generate_weights(self):

        self.kernel = tf.clip_by_value(self.kernel, 0.0, 1.0)

        return tf.map_fn(lambda p: bitstream_generator_exact(p, self.bit_size), self.kernel)

    def call(self, inputs):
        
        w = self.generate_weights()

        def compute_mult(weight, inputs):
            # cast to and-able form
            weight = tf.cast(weight, dtype=tf.int16)
            inputs = tf.cast(inputs, dtype=tf.int16)

            # "multiply" inputs and weight
            temp = weight & inputs

            # "sum" values together
            temp = tf.reduce_sum(temp, axis=0)

            # normalise values
            #temp = tf.cond(temp > 0, lambda: 1, lambda: 0)

            return temp

            # return tf.constant(temp, dtype=tf.float32)


        def compute_output(weight, inputs):
            # itterate through output size
            return tf.map_fn(lambda w: compute_mult(w, inputs), weight, dtype=tf.float32)

        output = None
        
        # if batch data then unpack
        if inputs.get_shape().ndims == 3:
            output = tf.map_fn(lambda x: compute_output(w, x), inputs)
        else:
            output = compute_output(w, inputs)

        print("Shape: ", output[output > 0])
        #output = tf.math.reduce_sum(output, axis=tf.rank(output) - 1) / self.bit_size
        output = output[output > 0] / self.bit_size
        print("Output ", output)
        return tf.zeros((1))
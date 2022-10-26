import tensorflow as tf
from tensorflow.keras.layers import Layer


class BitOutput(Layer):
    def __init__(self):
        super(BitOutput, self).__init__()

    def call(self, input):

        # convert to numpy array
        x = tf.get_static_value(input)

        # count number of 1s and divide by bitlength
        x = (x == 1).sum(axis=tf.rank(input) - 1) / input.shape[-1]
        
        return tf.constant(x)


def bitoutput_test():

    layer = BitOutput()

    # test 3d to 2d
    x1 = tf.constant([
        [[1,1,0,0], [1,0,0,0], [1,0,0,0]],
        [[0,1,0,0], [1,0,0,1], [1,0,0,0]],
    ])
    y1 = layer(x1)
    print("3D to 2D Convert")
    print(x1)
    print(y1)

    # test 2d to 1d
    x2 = tf.constant([
        [1,1,0,0], [1,0,0,0], [1,0,0,0]
    ])
    y2 = layer(x2)
    print("\n2D to 1D Convert")
    print(x2)
    print(y2)

if __name__ == "__main__":
    bitoutput_test()
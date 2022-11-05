import numpy as np


'''
input = [
    [0,1,1,0,0,0,1,0],          bitstream A
    [1,0,1,0,1,0,0,0]           bitstream B
]
'''


# constants
bN = 16
wN = 16


'''
generates bitstream of length N with probability p
'''
def bitstream_generator(p, N):
    return np.random.binomial(1, p, N)


'''
generates bitstream of length N with a better result than using binomial
'''
def bitstream_generator_exact(p, N):
    n_bits = int(np.round(p * N))
    bs = np.concatenate((
        np.ones((n_bits)),
        np.zeros((N - n_bits))
    ))
    return np.random.permutation(bs)


'''
takes in numpy array and converts to floating value [0,1]
'''
def bitstream_integrator(bs):
    return (bs == 1).sum() / len(bs)


def bitstream_mux_or(a, b):
    mux_mask = np.tile([0, 1], len(a) // 2)
    mux_mask_n = (mux_mask == 0)

    return (((a == 1) & (mux_mask == 1)) | ((b == 1) & (mux_mask_n == 1))) * 1

def bitstream_mux_sum(a):
    input_count = a.shape[0]
    input_length = a.shape[1]

    y = np.zeros(input_length)

    for i in range(0, input_length):
        y[i] = a[i % input_count][i]

    return y


class Neuron:
    def __init__(self, inputSize, bsLength, bwLength):
        self.input_length = bsLength
        self.weight_length = bwLength
        
        self.weights = np.zeros(shape=(inputSize, bwLength), dtype=np.int32)


    def set_weights(self, w):
        for i in range(len(w)):
            self.weights[i] = np.array(bitstream_generator_exact(w[i], self.weight_length))

    def increment_weights(self, l: np.ndarray):
        for i in range(len(self.weights)):
            self.weights[i] = np.concatenate((self.weights[:l], self.weights[l:]))


    def call(self, input):

        # bitstream mulitplication
        x = input & self.weights[:,:self.input_length]

        # sum columns
        x = x.sum(axis=0)

        # normalise values
        x = (x > 0) * 1

        return x


class RealNeuron:

    def __init__(self, weights):
        self.weights = weights

    def call(self, input):
        x = input * self.weights
        return sum(x)
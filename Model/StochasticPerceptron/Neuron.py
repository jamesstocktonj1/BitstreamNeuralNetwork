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


class Neuron:
    def __init__(self, inputSize, bsLength, bwLength):
        self.input_length = bsLength
        self.weight_length = bwLength
        
        self.weights = np.zeros(shape=(inputSize, bwLength), dtype=np.int32)


    def set_weights(self, w):
        for i in range(len(w)):
            if (w[i] < 0) or (w[i] > 1):
                print("Error: Non-Positive Weight, value: {}".format(w[i]))
                self.weights[i] = np.zeros(self.weight_length)
            else:
                self.weights[i] = np.array(bitstream_generator_exact(w[i], self.weight_length))

    def increment_weights(self, l: np.ndarray):
        for i in range(len(self.weights)):
            self.weights[i] = np.concatenate((self.weights[:l], self.weights[l:]))


    def call(self, input):

        # bitstream mulitplication
        x = (input == 1) & (self.weights[:,:self.input_length] == 1)

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
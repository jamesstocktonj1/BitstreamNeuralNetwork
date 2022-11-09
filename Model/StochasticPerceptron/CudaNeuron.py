import cupy as np


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
    n_bits = np.sum(np.round_(p * N))
    #print("{} x {} = {}".format(p, N, n_bits))
    n_bits = int(n_bits)
    bs = np.concatenate((
        np.ones((n_bits)),
        np.zeros((N - n_bits))
    ))
    return np.random.permutation(bs)


'''
takes in numpy array and converts to floating value [0,1]
'''
def bitstream_integrator(bs):
    return np.sum(bs == 1) / len(bs)


class Neuron:
    def __init__(self, inputSize, bsLength, bwLength):
        self.input_length = bsLength
        self.weight_length = bwLength
        
        self.weights = np.zeros(shape=(inputSize, bwLength), dtype=np.int32)


    def set_weights(self, w):
        for i in range(len(w)):
            self.weights[i] = np.array(bitstream_generator_exact(w[i], self.weight_length))

    def increment_weights(self, l):
        for i in range(len(self.weights)):
            self.weights[i] = np.concatenate((self.weights[:l], self.weights[l:]))


    def call(self, input):

        input = np.array(input)

        # bitstream mulitplication
        x = np.bitwise_and(input == 1, self.weights[:,:self.input_length] == 1)

        # sum columns
        x = np.sum(x, axis=0)

        # normalise values
        x = (x > 0) * 1

        return np.asnumpy(x)


class RealNeuron:

    def __init__(self, weights):
        self.weights = weights

    def call(self, input):
        x = input * self.weights
        return sum(x)
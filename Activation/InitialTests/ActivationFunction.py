import numpy as np


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


def bit_and(a, b):
    if (a == b) and (a == 1):
        return True
    else:
        return False

def bit_or(a, b):
    if a == 1:
        return True
    elif b == 1:
        return True
    else:
        return False



class StreamBuffer:
    def __init__(self, bit_length):
        self.bit_length = bit_length

        self.buffer = np.zeros((bit_length))

    def get_tap(self, index):
        return self.buffer[index]

    def next(self, value=0):
        self.buffer = np.concatenate(([value], self.buffer[:-1]))


def test_stream_buffer():
    N = 16
    test = bitstream_generator_exact(0.7, N)

    buf = StreamBuffer(N)
    for n in test[::-1]:
        buf.next(n)

    print("Stream Buffer Test")
    print(test)
    print(buf.buffer)


if __name__ == "__main__":
    test_stream_buffer()
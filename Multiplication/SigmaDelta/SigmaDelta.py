import random


def bitAND(a, b):
    return (a == '1') and (b == '1')

def streamAND(a, b):
    z = ""
    for x, y in zip(a, b):
        if bitAND(x, y):
            z += "1"
        else:
            z += "0"

    return z

def streamProbability(stream):
    return stream.count('1') / len(stream)

class SigmaDeltaStream:

    def __init__(self, value, bitdepth):
        bitCount = int(value * bitdepth)
        bitStream = "0"

        for i in range((2 ** bitdepth) - 1):
            if streamProbability(bitStream) < value:
                bitStream += '1'
            else:
                bitStream += '0'

        self.curIndex = 0
        self.bitStream = bitStream
    
    def randomise(self):
        newStream = ""
        oldStream = list(self.bitStream)
        lenStore = len(self.bitStream) + 0

        for i in range(lenStore):
            newStream += oldStream.pop(random.randint(0, len(oldStream) - 1))

        self.bitStream = newStream

    def next(self):
        self.curIndex = (self.curIndex + 1) % len(self.bitStream)
        return self.bitStream[self.curIndex]
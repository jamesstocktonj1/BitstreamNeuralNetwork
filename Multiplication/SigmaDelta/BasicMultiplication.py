from SigmaDelta import *
import matplotlib.pyplot as plt

# constants
bitRange = 16
probRange = 10
N = 8

p1 = 0.25
p2 = 0.50
p = p1 * p2

# create bitstreams and circular weight buffer
streamA = SigmaDeltaStream(p1, N)
streamB = SigmaDeltaStream(p2, N)
weightB = SigmaDeltaStream(p2, N + 1)


# mean and variance stats
ssP = 0
ssV = 0

swP = 0
swV = 0

for i in range(probRange):

    streamA.randomise()
    streamB.randomise()
    weightB.randomise()


    temp = ""
    for j in range(N * N):
        if bitAND(streamA.get(), streamB.get()):
            temp += "1"
        else:
            temp += "0"

    ssP += streamProbability(temp)
    ssV += ((p - streamProbability(temp)) ** 2)


    streamA.curIndex = 0
    temp = ""
    for j in range(N * (N + 1)):
        if bitAND(streamA.get(), weightB.get()):
            temp += "1"
        else:
            temp += "0"

    swP += streamProbability(temp)
    swV += ((p - streamProbability(temp)) ** 2)


ssP = ssP / probRange
swP = swP / probRange

ssV = (ssV / probRange) ** 0.5
swV = (swV / probRange) ** 0.5

print("Results\n")
print("Stream * Stream")
print("Mean: {:.4f}\tVariance:{:.5f}\n".format(ssP, (ssV / probRange) ** 0.5))
print("Stream * Weight")
print("Mean: {:.4f}\tVariance:{:.5f}\n".format(swP, (swV / probRange) ** 0.5))
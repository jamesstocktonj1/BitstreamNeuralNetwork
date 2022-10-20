from SigmaDelta import *
import matplotlib.pyplot as plt

# constants
bitRange = 16
probRange = 128
N = 16

p1 = 0.34
p2 = 0.50
p = p1 * p2

# create bitstreams and circular weight buffer
streamA = SigmaDeltaStream(p1, N)
streamB = SigmaDeltaStream(p2, N)
weightB = SigmaDeltaStream(p2, N + 1)


# mean and variance stats
ssP = 0
ssV = 0
ssM = []

swP = 0
swV = 0
swM = []


for i in range(probRange):

    streamA.randomise()
    streamB.randomise()
    weightB.randomise()


    temp = ""
    for j in range(N * 2):
        if bitAND(streamA.get(), streamB.get()):
            temp += "1"
        else:
            temp += "0"
        streamA.next()
        streamB.next()

    ssP += streamProbability(temp)
    ssV += ((p - streamProbability(temp)) ** 2)
    ssM.append(streamProbability(temp))


    streamA.curIndex = 0
    temp = ""
    for j in range(N * 2):
        if bitAND(streamA.get(), weightB.get()):
            temp += "1"
        else:
            temp += "0"
        streamA.next()
        weightB.next()

    swP += streamProbability(temp)
    swV += ((p - streamProbability(temp)) ** 2)
    swM.append(streamProbability(temp))


ssP = ssP / probRange
swP = swP / probRange

ssV = (ssV / probRange) ** 0.5
swV = (swV / probRange) ** 0.5

print("Real Multiplication")
print("Mean: {:.4f}\n".format(p))
print("Stream * Stream")
print("Mean: {:.4f}\tVariance:{:.5f}\n".format(ssP, (ssV / probRange) ** 0.5))
print("Stream * Weight")
print("Mean: {:.4f}\tVariance:{:.5f}\n".format(swP, (swV / probRange) ** 0.5))


'''
fig1 = plt.subplot(2, 1, 1)
#fig1.set_xlim([0, 1])
fig1.hist(ssM, density=2)

fig2 = plt.subplot(2, 1, 2)
#fig2.set_xlim([0, 1])
fig2.hist(swM, density=2)
'''
plt.hist([ssM, swM], bins=15, alpha=0.7, color=['green', 'blue'], label=['Stream * Stream', 'Stream * Weight'])
plt.axvline(p, color='red', linestyle='dashed', linewidth=1)
plt.legend(prop={'size': 10})
plt.title('Stochastic Multiplication Distribution')

plt.show()
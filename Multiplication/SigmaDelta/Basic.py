from SigmaDelta import *
import matplotlib.pyplot as plt


bitRange = 16
probRange = 20

p1 = 0.25
p2 = 0.50


mean = []
vari = []

for bitSize in range(1, bitRange):

    streamA = SigmaDeltaStream(p1, bitSize)
    streamB = SigmaDeltaStream(p2, bitSize)

    mSum = 0
    vSum = 0
    for i in range(probRange):
        streamA.randomise()
        streamB.randomise()

        s = streamAND(streamA.bitStream, streamB.bitStream)
        p = streamProbability(s)
        mSum += p
        vSum += (p - (p1 * p2))**2

    mean.append(mSum / probRange)
    vari.append((vSum / probRange) ** 0.5)


fig1 = plt.subplot(2, 1, 1)
fig1.plot(mean)
fig1.set_ylabel("Mean")

fig2 = plt.subplot(2, 1, 2)
fig2.plot(vari)
fig2.set_ylabel("Variance")
fig2.set_xlabel("Bit Length")

#plt.show()
plt.savefig("SigmaDelta_meanVar.png")
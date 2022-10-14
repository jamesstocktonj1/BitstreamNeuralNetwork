import matplotlib.pyplot as plt
import numpy as np

wave = [
    0.0000, 0.0980,
    0.1951, 0.2903,
    0.3827, 0.4714,
    0.5556, 0.6344,
    0.7071, 0.7730,
    0.8315, 0.8819,
    0.9239, 0.9569,
    0.9808, 0.9952,
    1.0000, 0.9952,
    0.9808, 0.9569,
    0.9239, 0.8819,
    0.8315, 0.7730,
    0.7071, 0.6344,
    0.5556, 0.4714,
    0.3827, 0.2903,
    0.1951, 0.0980,
]


def delta_modulator(data, quality):
    temp = []
    curPoint = 0

    for d in data:
        if d > curPoint:
            temp.append(1)
            curPoint += quality
        else:
            temp.append(0)
            curPoint -= quality
    
    return temp


def delta_demodulator(data, quality):
    temp = []
    curPoint = 0

    for d in data:
        if d > 0:
            curPoint += quality
        else:
            curPoint -= quality
        temp.append(curPoint)
    
    return temp



modData = delta_modulator(wave, 0.2)
demodData = delta_demodulator(modData, 0.2)


fig1 = plt.subplot(2, 1, 1)
fig1.plot(wave)
fig1.plot(demodData)
fig1.set_ylabel("Waveform")

fig2 = plt.subplot(2, 1, 2)
fig2.step(range(len(modData)), modData)
fig2.set_xlabel("Time")
fig2.set_ylabel("Delta Bitstream")
#plt.show()
plt.savefig("data/delta_waveform.png")
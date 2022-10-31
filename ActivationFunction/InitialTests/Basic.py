import numpy as np
import matplotlib.pyplot as plt

from ActivationFunction import *



N = 1024
tap_index = (N // 2)



buffer = StreamBuffer(N)
half_buf = StreamBuffer(N)


fig1 = plt.subplot(1, 2, 1)
fig1.set_title("Activation Function")
fig1.set_aspect("equal", adjustable="box")
fig1.set_xlim([0, 1])
fig1.set_ylim([0, 1])

fig2 = plt.subplot(1, 2, 2)
fig2.set_title("Sigmoid")
fig2.set_aspect("equal", adjustable="box")
fig2.set_xlim([0, 1])
fig2.set_ylim([0, 1])


for i in range(50):
    x = bitstream_generator_exact(i/50, N)
    y = np.zeros((N))

    for j in range(N):
        buffer.next(x[j])
        y[j] = bit_or(buffer.get_tap(0), buffer.get_tap(tap_index))

    y = bitstream_integrator(y)

    print("{}, {}".format(i/50, y))

    fig1.plot(i/50, y, 'ro')
    fig2.plot(i/50, np.tanh(i/50), 'bo')


plt.suptitle("{}-Bit AND Activation Function".format(N))
plt.savefig("images/activation_tanh_better_or_{}bit_{}tap.png".format(N, tap_index))
#plt.show()
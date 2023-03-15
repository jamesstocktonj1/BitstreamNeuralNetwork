import numpy as np
import matplotlib.pyplot as plt

from LFSR import LFSR


X = 16


def plot_phase():
    bs1 = np.zeros(X)
    bs2 = np.zeros(X)

    sr1 = LFSR(5)
    sr2 = LFSR(2)

    for i in range(X):
        bs1[i] = (sr1.get() < 128) * 1
        bs2[i] = (sr2.get() < 128) * 1

        sr1.shift()
        sr2.shift()



    plt.figure()

    fig1 = plt.subplot(2, 1, 1)
    fig2 = plt.subplot(2, 1, 2)

    fig1.step(np.arange(X), bs1)
    fig2.step(np.arange(X), bs2)

    fig1.set_ylabel("Bitstream A")
    fig2.set_ylabel("Bitstream B")

    plt.savefig("images/phase_plot.png")
    plt.close()

if __name__ == "__main__":
    plot_phase()
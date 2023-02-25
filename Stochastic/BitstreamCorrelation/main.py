import numpy as np
import matplotlib.pyplot as plt

from LFSR import LFSR






def generatre_bitstream_matrix():

    sr = LFSR(0)

    bitMat = np.zeros((256, 255))

    # itterate through starting values
    for i in range(256):
        sr.reg = i

        # itterate through bitstream length
        for j in range(255):
            bitMat[i,j] = (sr.get() < 129) * 1
            sr.shift()

    return bitMat

def compare_correlation(bitstreamMatrix):

    corMatrix = np.zeros((256, 256))

    for i in range(256):

        for j in range(256):
            # xnor the two bitstreams
            cor = (bitstreamMatrix[i] == bitstreamMatrix[j]) * 1
            corMatrix[i,j] = cor.sum() / cor.size

            if i == j:
                corMatrix[i,j] = 0.5

    return corMatrix


def plot_correlation_matrix(correlationMatrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(correlationMatrix, cmap=plt.cm.RdYlGn.reversed())

    fig.colorbar(cax)

    plt.title("Bitstream Correlation (8-bit LFSR)")
    plt.savefig("images/bitstream_correlation.png")
    plt.show()


def main():
    correlationMatrix = np.random.randn(5, 5)

    bitstreamMatrix = generatre_bitstream_matrix()
    correlationMatrix = compare_correlation(bitstreamMatrix)

    corMax = correlationMatrix.max()
    corMin = correlationMatrix.min()

    corAvr = correlationMatrix.sum() / (256 * 256)

    print("Largest Positive Correlation: {}".format(corMax))
    print("Largest Negative Correlation: {}".format(corMin))
    print("Average Correlation: {}".format(corAvr))

    plot_correlation_matrix(correlationMatrix)


if __name__ == "__main__":
    main()
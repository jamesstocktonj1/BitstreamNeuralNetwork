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

    X = 256

    corMatrix = np.zeros((X, X))

    for i in range(X):

        for j in range(X):
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

def plot_compatability(correlationMatrix):
    fig = plt.figure()

    corAvg = (correlationMatrix ** 2).sum(axis=1) / correlationMatrix.shape[0]

    print(corAvg.min())
    print(corAvg.max())

    plt.bar(np.arange(0, correlationMatrix.shape[0]), corAvg)

    plt.savefig("images/bitstream_seed_quality.png")
    plt.show()


def main():
    correlationMatrix = np.random.randn(5, 5)

    bitstreamMatrix = generatre_bitstream_matrix()
    correlationMatrix = compare_correlation(bitstreamMatrix)

    corMax = (correlationMatrix.max() - 0.5)
    corMin = (correlationMatrix.min() - 0.5)

    corAvr = ((correlationMatrix.sum() / (256 * 256)) - 0.5)

    print("Largest Positive Correlation: {}".format(corMax))
    print("Largest Negative Correlation: {}".format(corMin))
    print("Average Correlation: {}".format(corAvr))

    plot_compatability(correlationMatrix)
    plot_correlation_matrix(correlationMatrix)


if __name__ == "__main__":
    main()
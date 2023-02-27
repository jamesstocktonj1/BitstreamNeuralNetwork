from LFSR import LFSR, LFSR16
import matplotlib.pyplot as plt
import numpy as np


N = 255


def split_bitstream(bs, split):
    N = bs.size // split

    split_bs = np.zeros((split, N))

    for i in range(split):
        split_bs[i] = bs[N*i:N*(i+1)]

    return split_bs


def main():
    sr = LFSR(0b11010)
    bs = []

    value = 184
    expected = 0.72157

    for i in range(N):
        sr.shift()
        bs.append((sr.get() < value) * 1)

    bs = np.array(bs)

    plt.figure()
    mse = []

    for split_count in range(1, 16):
        #split_count = 8

        split_bs = split_bitstream(bs, split_count)
        split_values = np.zeros(split_count)

        print("\nSplit Count: {}".format(split_count))

        for i, b in enumerate(split_bs):
            split_values[i] = b.sum() / b.size
            print("{:.5f}, ".format(b.sum() / b.size), end="")

        mse.append(((expected - split_values) ** 2).sum())

        print("\nMean Square Error: {:.5f}".format(((expected - split_values) ** 2).sum()))

    plt.xlabel("Number of Splits (255 / x)")
    plt.ylabel("Mean Square Error")
    plt.plot(list(range(1, 16)), mse)
    plt.savefig("images/split_error.png")
    plt.show()

    print("Bitstream Value: {}".format(bs.sum() / bs.size))
    print("Expected Value:  {}".format(value / N))




if __name__ == "__main__":
    main()
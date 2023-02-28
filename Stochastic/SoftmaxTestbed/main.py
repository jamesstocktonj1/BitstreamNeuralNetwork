import numpy as np


def bs_sum(x):
    return 1 - np.product(1 - x)


def softmax(x):
    y = np.zeros((x.shape[-1]))
    x_e = np.exp(-1 * (1 - x))

    for i in range(x.shape[-1]):

        e_sum = 1
        for j in range(x.shape[-1]):
            if i != j:
                e_sum *= (1 - x[j])

        e_sum = 1 - e_sum

        y[i] = x_e[i] / (x_e[i] + e_sum)

    return y * 0.8


def main():

    x = np.random.randint(0,100, size=(5)) / 100

    xe = np.exp(x)
    y = xe / xe.sum()


    xe_hat = np.exp(-1 * (1 - x))
    y_hat = xe_hat / xe_hat.sum()
    
    y_hat_hat = softmax(x)

    print("Input:   {}".format(x))
    print("Softmax: {}\t{}".format(y, y.sum()))
    print("Testmax: {}\t{}".format(y_hat, y_hat.sum()))
    print("Testmax: {}\t{}".format(y_hat_hat, y_hat_hat.sum()))




if __name__ == "__main__":
    main()
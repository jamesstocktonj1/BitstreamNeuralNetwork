import numpy as np


def bs_sum(x):
    return 1 - np.product(1 - x)




def main():

    x = np.random.randint(0,100, size=(3)) / 100

    xe = np.exp(x)
    y = xe / xe.sum()


    xe_hat = np.exp(-1 * (1 - x))
    y_hat = xe_hat / xe_hat.sum()
    
    y_hat_hat = np.zeros((3))
    for i in range(3):
        den = 1
        for j in range(3):
            if i != j:
                den *= (1 - xe_hat[j])
        den = 1 - den
        y_hat_hat[i] = xe_hat[i] / (xe_hat[i] + den)
    y_hat_hat *= 0.8
    # y_hat_hat = 1 - y_hat_hat

    print("Input:   {}".format(x))
    print("Softmax: {}\t{}".format(y, y.sum()))
    print("Testmax: {}\t{}".format(y_hat, y_hat.sum()))
    print("Testmax: {}\t{}".format(y_hat_hat, y_hat_hat.sum()))




if __name__ == "__main__":
    main()
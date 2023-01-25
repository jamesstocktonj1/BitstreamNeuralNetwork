import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm


X = 50


def loss_function(y, x, w):

    # y_hat = (x[0] * w[0]) + (x[1] * w[1]) - (x[0] * w[0] * x[1] * w[1])
    # y_hat = 1 / (1 + np.exp2(-1 * y_hat))

    # return (y - y_hat) ** 2

    return (y - 1 + ((1 - (x[0]*w[0])) * (1 - (x[1]*w[1])))) ** 2



def plot_loss():
    fig = plt.figure()
    fig1 = fig.add_subplot(projection='3d')

    C = [
        [1, -0.78], 
        [-0.78, 1]
    ]
    A = la.cholesky(C)
    xa = np.random.randn(X//2, 2)
    xa = np.dot(xa, A)

    xb = np.random.randn(X//2, 2)
    xb = np.dot(xb, A) + 2

    x_hat = np.vstack([xa, xb])
    x_hat = np.interp(x_hat, (x_hat.min(), x_hat.max()), (0, 1))
    y_hat = np.hstack([np.zeros(X//2), np.ones(X//2)]).reshape(-1, 1)

    x = np.arange(0, 1, 1/X)
    y = np.arange(0, 1, 1/X)
    x1, x2 = np.meshgrid(x, y)

    loss = np.zeros((X, X))

    for i in range(X):
        for j in range(X):
            
            lossSum = 0
            for k in range(X):
                
                lossSum += loss_function(y_hat[k], x_hat[k], [x1[i,j], x2[i,j]])

            loss[i,j] = lossSum / X

    surf = fig1.plot_surface(x1, x2, loss, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf)
    plt.show()

if __name__ == "__main__":
    plot_loss()
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from NeuralModel import SimpleModel


X = 250


# create linearly seperable dataset
C = [
    [1, -0.78], 
    [-0.78, 1]
]
A = la.cholesky(C)
xa = np.random.randn(X//2, 2)
xa = np.dot(xa, A)

xb = np.random.randn(X//2, 2)
xb = np.dot(xb, A) + 2

x = np.vstack([xa, xb])
x = np.interp(x, (x.min(), x.max()), (0, 1))
y = np.hstack([np.zeros(X//2), np.ones(X//2)]).reshape(-1, 1)



def training_loop(x, y):
    model = SimpleModel(0.000125)

    # model.layer1.weights = np.random.randn(2, 2)
    # model.layer1.weights = np.interp(model.layer1.weights, (model.layer1.weights.min(), model.layer1.weights.max()), (0.4, 0.6))

    # model.layer2.weights = np.random.randn(1,2)
    # model.layer2.weights = np.interp(model.layer2.weights, (model.layer2.weights.min(), model.layer2.weights.max()), (0.4, 0.6))

    # correct = np.zeros(y.shape)
    model.init_weights()
    
    # for e in range(2):

    #     # initial correct points
    #     for rxy in range(len(x)):
    #         y_hat = model.call(x[rxy])

    #         if (y_hat > 0.5) == (y[rxy] == 1):
    #             correct[rxy] = 1

    #     print("Pass: {}\tIncorrect Points: {}".format(e, np.sum(correct == 0)))

    #     while np.sum(correct) < len(y):
    #         rxy = np.random.choice(np.where(correct < 1)[0])
            
    #         grad1, grad2 = model.grad(x[rxy], y[rxy])
    #         print(grad1, grad2)

    #         # run through neuron
    #         y_hat = model.call(x[rxy])

    #         # if correctly classified
    #         if (y_hat > 0.5) == (y[rxy] == 1):
    #             correct[rxy] = 1

    for e in range(10):
        for rxy in range(len(x)):
            grad1, grad2 = model.grad(x[rxy], y[rxy])
            print(model.layer1.weights)
            print(model.layer2.weights)
    
    # evaluate model
    correct = np.zeros(y.shape)
    for rxy in range(len(x)):
        y_hat = model.call(x[rxy])

        if (y_hat > 0.5) == (y[rxy] == 1):
            correct[rxy] = 1

    print("Model Evaluation, Accuracy: {:.2f}%".format((np.sum(correct == 1) / len(correct)) * 100))



if __name__ == "__main__":
    training_loop(x, y)
import numpy as np





class NeuralLayer:

    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.input_size = input_size
        self.output_size = output_size

        self.no_activation = False

    def init_weights(self, u):
        self.weights = 0.5 + u * np.random.randn(self.output_size, self.input_size)


    def grad_loss(self, x, y):
        L = y - 1 + np.product(1 - (self.weights * x), axis=1)
        L = L.reshape(L.size, 1)

        b = np.product(1 - (self.weights * x), axis=1)
        c = b.reshape(b.size, 1) * (1 / (1 - (self.weights * x)))

        z = 1 - np.product(1 - (self.weights * x), axis=1)
        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return -2 * L * self.weights * c

    def grad_layer(self, x, z):
        b = np.product(1 - (self.weights * x), axis=1)
        c = b.reshape(b.size, 1) * (1 / (1 - (self.weights * x)))

        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return -1 * self.weights * c * z

    def grad_weight(self, x):
        b = np.product(1 - (self.weights * x), axis=1)
        c = b.reshape(b.size, 1) * (1 / (1 - (self.weights * x)))

        z = 1 - np.product(1 - (self.weights * x), axis=1)
        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return -1 * x * c * z

    def activation(self, z):
        if self.no_activation:
            return z
        else:
            return 1 / (1 + np.exp(-8 * (z - 0.5)))

    def activation_grad(self, z):
        if self.no_activation:
            return np.ones((z.size))
        else:
            return self.activation(z) * (1 - self.activation(z))

    def call(self, x):
        z = 1 - np.product(1 - (self.weights * x), axis=1)
        return self.activation(z)







def neuron_layer_test():

    layer = NeuralLayer(2, 2)
    layer.no_activation = True

    # a = np.array([0.1, 0.2])
    # b = np.array([0.5, 0.7])
    # w = np.array([[0.1, 0.2], [0.3, 0.4]])

    a = np.random.randint(0, 10, size=2) / 10
    b = np.random.randint(0, 10, size=2) / 10
    w = np.array([
        np.random.randint(0, 10, size=2) / 10,
        np.random.randint(0, 10, size=2) / 10
    ])

    print("\n\nTest Data...")
    print("x: ", a)
    print("y: ", b)
    print("w: ", w)

    layer.weights = w

    grads = layer.grad_weight(a)

    exp_grads = np.zeros((2,2))
    exp_grads[0,0] = -1 * a[0] * (1 - (w[0,1] * a[1]))
    exp_grads[0,1] = -1 * a[1] * (1 - (w[0,0] * a[0]))
    exp_grads[1,0] = -1 * a[0] * (1 - (w[1,1] * a[1]))
    exp_grads[1,1] = -1 * a[1] * (1 - (w[1,0] * a[0]))

    print("\nTesting Grad Weight...")
    print("Expected: ", exp_grads)
    print("Got:      ", grads)

    grads = layer.grad_layer(a, b)

    exp_grads = np.zeros((2,2))
    exp_grads[0,0] = -1 * w[0,0] * (1 - (w[0,1] * a[1]))
    exp_grads[0,1] = -1 * w[0,1] * (1 - (w[0,0] * a[0]))
    exp_grads[1,0] = -1 * w[1,0] * (1 - (w[1,1] * a[1]))
    exp_grads[1,1] = -1 * w[1,1] * (1 - (w[1,0] * a[0]))

    print("\nTesting Grad Layer...")
    print("Expected: ", exp_grads)
    print("Got:      ", grads)

    grads = layer.grad_loss(a, b)

    exp_grads = np.zeros((2,2))
    exp_grads[0,0] = -2 * w[0,0] * (1 - (w[0,1] * a[1])) * (b[0] - 1 + ((1 - (w[0,0] * a[0])) * (1 - (w[0,1] * a[1]))))
    exp_grads[0,1] = -2 * w[0,1] * (1 - (w[0,0] * a[0])) * (b[0] - 1 + ((1 - (w[0,0] * a[0])) * (1 - (w[0,1] * a[1]))))
    exp_grads[1,0] = -2 * w[1,0] * (1 - (w[1,1] * a[1])) * (b[1] - 1 + ((1 - (w[1,0] * a[0])) * (1 - (w[1,1] * a[1]))))
    exp_grads[1,1] = -2 * w[1,1] * (1 - (w[1,0] * a[0])) * (b[1] - 1 + ((1 - (w[1,0] * a[0])) * (1 - (w[1,1] * a[1]))))

    print("\nTesting Grad Loss...")
    print("Expected: ", exp_grads)
    print("Got:      ", grads)


    y = layer.call(a)
    exp_y = np.zeros((2))
    exp_y[0] = 1 - ((1 -(w[0,0] * a[0])) * (1 - (w[0,1] * a[1])))
    exp_y[1] = 1 - ((1 -(w[1,0] * a[0])) * (1 - (w[1,1] * a[1])))

    print("\nTesting Call...")
    print("Expected: ", exp_y)
    print("Got:      ", y)

if __name__ == "__main__":
    neuron_layer_test()
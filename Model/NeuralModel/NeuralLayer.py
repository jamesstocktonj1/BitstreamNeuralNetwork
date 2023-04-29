import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-8 * (z - 0.5)))

def softmax(z):
    return np.exp(z) / np.exp(z).sum()

def relu(z):
    y = np.zeros(z.shape[0])
    y[np.where(0.5 < z)] = (1.2 * z[np.where(0.5 < z)]) - 0.6
    y[np.where(0.75 < z)] = (2.8 * z[np.where(0.75 < z)]) - 1.8
    return y

def relu_grad(z):
    dy = np.zeros(z.shape[0])
    dy[np.where(0.5 < z)] = 1.2
    dy[np.where(0.75 < z)] = 2.8
    return dy

class NeuralLayer:

    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.input_size = input_size
        self.output_size = output_size

        self.no_activation = False
        self.softmax = False
        self.crossentropy = False
        self.relu = False

    def init_weights(self, u):
        self.weights = 0.5 + u * np.random.randn(self.output_size, self.input_size)

    def init_xavier(self, u):
        std = u * np.sqrt(2 / (self.input_size + self.output_size))
        self.weights = np.abs(std * np.random.randn(self.output_size, self.input_size)) 

    def grad_loss(self, z, y):
        L = y - z

        return (-2 * L).reshape((1, -1))

    def grad_layer(self, x, z):
        
        def else_function(a):
            def everything_else(b):
                return np.product(a[np.where(a != b)])

            f = np.vectorize(everything_else)
            return f(a)

        c = np.apply_along_axis(else_function, 1, 1 - (self.weights * x))

        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return self.weights * c * z

    def grad_weight(self, x):

        def else_function(a):
            def everything_else(b):
                return np.product(a[np.where(a != b)])

            f = np.vectorize(everything_else)
            return f(a)

        c = np.apply_along_axis(else_function, 1, 1 - (self.weights * x))

        z = 1 - np.product(1 - (self.weights * x), axis=1)
        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return x * c * z

    def activation(self, z):
        if self.no_activation:
            return z
        elif self.relu:
            return relu(z)
        elif self.softmax:
            return softmax(z)
        else:
            return sigmoid(z)

    def activation_grad(self, z):
        if self.no_activation:
            return np.ones((z.size))
        elif self.relu:
            return relu_grad(z)
        elif self.softmax:
            return softmax(z) * (1 - softmax(z))
        else:
            return sigmoid(z) * (1 - sigmoid(z))

    def call(self, x):
        z = 1 - np.product(1 - (self.weights * x), axis=1)
        return self.activation(z)


class NeuralLayerBias:

    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.input_size = input_size
        self.output_size = output_size

        self.no_activation = False
        self.softmax = False
        self.crossentropy = False
        self.relu = False

    def init_weights(self, u):
        self.weights = 0.5 + u * np.random.randn(self.output_size, self.input_size)
        self.bias = 0.5 + u * np.random.randn(self.output_size, 1)

    def init_xavier(self, u):
        std = u * np.sqrt(2 / (self.input_size + self.output_size))
        self.weights = np.abs(std * np.random.randn(self.output_size, self.input_size))
        self.bias = np.abs(std * np.random.randn(self.output_size, 1))

    def init_uniform(self, u):
        self.weights = np.random.rand(self.output_size, self.input_size) * u
        self.bias = np.random.rand(self.output_size, 1) * u

    def grad_loss(self, z, y):
        L = y - z

        return (-2 * L).reshape((1,-1))

    def grad_layer(self, x, z):
        
        def else_function(a):
            def everything_else(b):
                return np.product(a[np.where(a != b)])

            f = np.vectorize(everything_else)
            return f(a)

        c = np.apply_along_axis(else_function, 1, 1 - (self.weights * x))
        c = c * (1 - self.bias)

        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return self.weights * c * z

    def grad_weight(self, x):

        def else_function(a):
            def everything_else(b):
                return np.product(a[np.where(a != b)])

            f = np.vectorize(everything_else)
            return f(a)

        c = np.apply_along_axis(else_function, 1, 1 - (self.weights * x))
        c = c * (1 - self.bias)

        z = 1 - np.product(1 - (self.weights * x), axis=1)
        z = self.activation_grad(z)
        z = z.reshape(z.size, 1)

        return x * c * z

    def grad_bias(self, x):
        return np.product(1 - (self.weights * x), axis=1).reshape((-1, 1))

    def activation(self, z):
        if self.no_activation:
            return z
        elif self.relu:
            return relu(z)
        elif self.softmax:
            return softmax(z)
        else:
            return sigmoid(z)

    def activation_grad(self, z):
        if self.no_activation:
            return np.ones((z.size))
        elif self.relu:
            return relu_grad(z)
        elif self.softmax:
            return softmax(z) * (1 - softmax(z))
        else:
            return sigmoid(z) * (1 - sigmoid(z))

    def call(self, x):
        z = np.product(np.hstack((1 - self.bias, 1 - (self.weights * x))), axis=1)
        return self.activation(1-z)




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

    exp_grads = np.zeros((2))
    # exp_grads[0,0] = -2 * w[0,0] * (1 - (w[0,1] * a[1])) * (b[0] - 1 + ((1 - (w[0,0] * a[0])) * (1 - (w[0,1] * a[1]))))
    # exp_grads[0,1] = -2 * w[0,1] * (1 - (w[0,0] * a[0])) * (b[0] - 1 + ((1 - (w[0,0] * a[0])) * (1 - (w[0,1] * a[1]))))
    # exp_grads[1,0] = -2 * w[1,0] * (1 - (w[1,1] * a[1])) * (b[1] - 1 + ((1 - (w[1,0] * a[0])) * (1 - (w[1,1] * a[1]))))
    # exp_grads[1,1] = -2 * w[1,1] * (1 - (w[1,0] * a[0])) * (b[1] - 1 + ((1 - (w[1,0] * a[0])) * (1 - (w[1,1] * a[1]))))

    exp_grads[0] = -2 * (a[0] - b[0])
    exp_grads[1] = -2 * (a[1] - b[1])

    print("\nTesting Grad Loss...")
    print("Expected: ", exp_grads)
    print("Got:      ", grads)


    y = layer.call(a)
    exp_y = np.zeros((1,2))
    exp_y[0,0] = 1 - ((1 -(w[0,0] * a[0])) * (1 - (w[0,1] * a[1])))
    exp_y[0,1] = 1 - ((1 -(w[1,0] * a[0])) * (1 - (w[1,1] * a[1])))

    print("\nTesting Call...")
    print("Expected: ", exp_y)
    print("Got:      ", y)


def neuron_layer_complex_test():

    layer = NeuralLayer(3, 2)
    layer.no_activation = True
    
    x = np.random.randint(0, 10, size=3) / 10
    y = np.random.randint(0, 10, size=2) / 10
    w = np.random.randint(0, 10, size=(2,3)) / 10

    layer.weights = w

    z = layer.call(x)

    # test loss grad
    exp_loss_grad = np.zeros((1, 2))
    loss_grad = layer.grad_loss(z, y)

    exp_loss_grad[0,0] = -2 * (y[0] - z[0])
    exp_loss_grad[0,1] = -2 * (y[1] - z[1])

    print("\nTesting Grad Loss...")
    print("Expected: ", exp_loss_grad)
    print("Got:      ", loss_grad)


    # test layer grad
    exp_layer_grad = np.zeros((2, 3))
    layer_grad = layer.grad_layer(x, z)

    exp_layer_grad[0,0] = w[0,0] * (1 - (x[1] * w[0,1])) * (1 - (x[2] * w[0,2]))
    exp_layer_grad[0,1] = w[0,1] * (1 - (x[0] * w[0,0])) * (1 - (x[2] * w[0,2]))
    exp_layer_grad[0,2] = w[0,2] * (1 - (x[0] * w[0,0])) * (1 - (x[1] * w[0,1]))

    exp_layer_grad[1,0] = w[1,0] * (1 - (x[1] * w[1,1])) * (1 - (x[2] * w[1,2]))
    exp_layer_grad[1,1] = w[1,1] * (1 - (x[0] * w[1,0])) * (1 - (x[2] * w[1,2]))
    exp_layer_grad[1,2] = w[1,2] * (1 - (x[0] * w[1,0])) * (1 - (x[1] * w[1,1]))

    print("\nTesting Grad Layer...")
    print("Expected: ", exp_layer_grad)
    print("Got:      ", layer_grad)


    # test weight grad
    exp_weight_grad = np.zeros((2, 3))
    weight_grad = layer.grad_weight(x)

    exp_weight_grad[0,0] = x[0] * (1 - (x[1] * w[0,1])) * (1 - (x[2] * w[0,2]))
    exp_weight_grad[0,1] = x[1] * (1 - (x[0] * w[0,0])) * (1 - (x[2] * w[0,2]))
    exp_weight_grad[0,2] = x[2] * (1 - (x[0] * w[0,0])) * (1 - (x[1] * w[0,1]))

    exp_weight_grad[1,0] = x[0] * (1 - (x[1] * w[1,1])) * (1 - (x[2] * w[1,2]))
    exp_weight_grad[1,1] = x[1] * (1 - (x[0] * w[1,0])) * (1 - (x[2] * w[1,2]))
    exp_weight_grad[1,2] = x[2] * (1 - (x[0] * w[1,0])) * (1 - (x[1] * w[1,1]))

    print("\nTesting Grad Weight...")
    print("Expected: ", exp_weight_grad)
    print("Got:      ", weight_grad)





if __name__ == "__main__":
    # neuron_layer_test()
    neuron_layer_complex_test()
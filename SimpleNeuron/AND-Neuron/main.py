import numpy as np
import matplotlib.pyplot as plt


weightsMap = [
    [[0.5, 0.2], [0.1, 0.6]],
    [[-0.2, 0.4]]
]

biasMap = [
    [],
    [0.5]
]


def activation_step(s):
    if s >= 0:
        return 1
    else:
        return 0

def acivation_signum(s):
    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0

def activation_relu(s):
    if s > 0:
        return s
    else:
        return 0

def activation_sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))


def neuron(weights, bias, inputs):
    
    # inputs * weights
    s = bias + 0
    for w, i in zip(weights, inputs):
        s += (w * i)

    # activation function
    return activation_relu(s)

def bit_neuron(weights, inputs):

    # inputs AND weights
    s = 0
    for w, i in zip(weights, inputs):
        if i == 1:
            s += w

    # activation function
    return activation_relu(s)


def network(inputs):
    memory = [inputs]

    for l in range(0, len(weightsMap)):
        memory.append([])

        # itterate through neuron in layer
        for n in range(0, len(weightsMap[l])):
            if l == 0:
                s = bit_neuron(weightsMap[l][n], memory[l])
            else:
                s = neuron(weightsMap[l][n], biasMap[l][n], memory[l])
            memory[l + 1].append(s)

    return memory

def print_memory(memory):

    print("Network State")
    for l in memory:
        print(l)



def main():

    inputs = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ]

    for i in inputs:
        out = network(i)
        print_memory(out)

    



if __name__ == "__main__":
    main()
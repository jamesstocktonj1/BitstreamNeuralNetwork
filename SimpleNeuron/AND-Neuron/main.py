import numpy as np
import matplotlib.pyplot as plt



weightsMap = [
    [[0.5, 0.2]]
]

inputsMap = [1, 1]

memory = []



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



def neuron(weights, inputs):

    # inputs AND weights
    s = 0
    for w, i in zip(weights, inputs):
        if i == 1:
            s += w

    # activation function
    return activation_relu(s)


def main():
    memory.append(inputsMap)

    print("Inputs: ", memory[0])

    # itterate through layers
    for l in range(0, len(weightsMap)):
        memory.append([])

        # itterate through neuron in layer
        for n in range(0, len(weightsMap[l])):
            memory[l + 1].append(neuron(weightsMap[l][n], memory[l]))


    print("Outputs: ", memory[-1])


if __name__ == "__main__":
    main()
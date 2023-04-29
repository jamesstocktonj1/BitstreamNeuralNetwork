import numpy as np
import sys, json


def print_array(arr):

    print("int arr [0:{}][0:{}] = '{{".format(arr.shape[0] - 1, arr.shape[1] - 1))
    for i in range(arr.shape[0]):

        print("\t'{", end="")
        for j in range(arr.shape[1]):
            print("{}".format(int(arr[i,j] * 256)), end="")

            if j != (arr.shape[1]-1):
                print(",", end="")
            else:
                print("}", end="")

        if i != (arr.shape[0]-1):
            print(",")

    print("\n};")



def main(filename):

    with open(filename, "r") as f:
        modelData = json.load(f)

    print("Loaded Model {}: {}".format(filename, modelData["info"]))

    print("Layer1: ")
    print_array(np.array(modelData["layer1"]))

    print("\nLayer2: ")
    print_array(np.array(modelData["layer2"]))

    print("\nLayer3: ")
    print_array(np.array(modelData["layer3"]))




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide input file")
    else:
        main(sys.argv[1])
import numpy as np
import json




def print_matrix(m):
    print(np.intc(np.round(m * 256)))



def load_model(name):

    with open(name, "r") as f:
        modelData = json.load(f)
    
    print("Loading Model Info: {}".format(modelData["info"]))

    l1_w = np.array(modelData["layer1"])
    l2_w = np.array(modelData["layer2"])
    l3_w = np.array(modelData["layer3"])

    l1_b = np.array(modelData["layer1b"])
    l2_b = np.array(modelData["layer2b"])
    l3_b = np.array(modelData["layer3b"])


    print("Weights 1")
    print_matrix(l1_w)

    print("Bias 1")
    print_matrix(l1_b)

    print("Weights 2")
    print_matrix(l2_w)

    print("Bias 2")
    print_matrix(l2_b)

    print("Weights 3")
    print_matrix(l3_w)

    print("Bias 3")
    print_matrix(l3_b)



if __name__ == "__main__":
    load_model("data/bias_model/best/best.json")
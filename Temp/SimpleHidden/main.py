import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


# parameters
N = list(2**n for n in range(4, 12))
X = 1000


# create normal dataset
x = np.vstack([np.random.randn(X//2, 2)/6 + 0.25, np.random.randn(X//2, 2)/6 + 0.75])
x = np.clip(x, 0, 1.0)
y = np.hstack([np.zeros(X//2), np.ones(X//2)]).reshape(-1, 1)


class PerceptronModel(Model):

    def __init__(self):
        super(PerceptronModel, self).__init__()

        self.input_layer = Input(shape=(2, ))
        self.norm = tf.keras.layers.Normalization(input_shape=(2, ), axis=None)
        #self.dense1 = Dense(2)
        self.dense2 = Dense(1, use_bias=False)

    def call(self, x):

        x = self.input_layer(x)
        x = self.norm(x)
        
        #x = self.dense1(x)
        x = self.dense2(x)
        
        return x

def train_model(x, y):

    #model = PerceptronModel()

    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2, )),
        tf.keras.layers.Normalization(input_shape=(2, ), axis=None),
        Dense(2, use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg()),
        Dense(1, use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg())
    ])
    

    a = tf.constant([0, 0])
    b = model(a)



    model.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    model.fit(x, y, epochs=50)

    #model.layers[2]

    print("Layer 1 Dense: ", model.layers[1].get_weights())
    print("Layer 2 Dense: ", model.layers[2].get_weights())
    #print("Layer 2 Dense: ", model.dense2.get_weights())

    model.summary()
    



def main():

    train_model(x, y)

if __name__ == "__main__":
    main()

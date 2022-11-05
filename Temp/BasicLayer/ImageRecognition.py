import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from Layer import BitLayer, BitCreator, BitIntegrator


# model parameters
N = 128

# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.input_layer = layers.Flatten()
        self.bit_layer = BitCreator(N)

        self.dense1 = BitLayer(128, N)
        self.dense2 = BitLayer(128, N)

        self.dense3 = BitLayer(10, N)
        self.output_layer = BitIntegrator()

    def call(self, x):
        print(x)
        x = self.input_layer(x)
        print(x)
        x = self.bit_layer(x)
        
        print(x)
        x = self.dense1(x)
        print(x)
        x = self.dense2(x)

        print(x)
        x = self.dense3(x)
        print(x)
        return self.output_layer(x)

model = MyModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(28, 28))
#model.fit(x_train, y_train, epochs=10)

'''
#model.summary()

# evaluate model
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss: {}\nAccuracy: {}".format(val_loss, val_acc))

model.save('basic.model')


# evaluate prediction
predictions = model.predict(x_test)

print("Prediction: {}".format(np.argmax(predictions[0])))
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.savefig('main.png')
#plt.show()
'''
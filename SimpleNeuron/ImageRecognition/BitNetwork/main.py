import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

from BitLayer import BitNeurons


# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# reshape from (len, 28, 28) to (len, 28 * 28)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# round any values greater than 0.0 to be equal to 1.0 (bit input)
round_threshold = 0.3
x_train = np.ceil(x_train - round_threshold)
x_test = np.ceil(x_test - round_threshold)


# create model
class BitModel(Model):

    def __init__(self):
        super(BitModel, self).__init__()

        self.input_layer = BitNeurons(128)
        self.dense1 = layers.Dense(128, activation=tf.nn.relu)

        self.softmax = layers.Dense(10, activation=tf.nn.softmax)

    def call(self, x, is_training=False):
        x = self.input_layer(x)

        x = self.dense1(x)

        if not is_training:
            x = self.softmax(x)

        return x


model = BitModel()

# train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# evaluate model
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss: {}\nAccuracy: {}".format(val_loss, val_acc))

# evaluate prediction
predictions = model.predict(x_test)

print("Prediction: {}".format(np.argmax(predictions[0])))
plt.imshow(x_test[0].reshape(28, 28), cmap=plt.cm.binary)
plt.savefig('main.png')
#plt.show()
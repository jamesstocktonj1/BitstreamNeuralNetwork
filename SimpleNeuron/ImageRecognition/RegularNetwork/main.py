import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt



# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# create model
class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.input_layer = layers.Flatten()

        self.dense1 = layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = layers.Dense(128, activation=tf.nn.relu)

        self.softmax = layers.Dense(10, activation=tf.nn.softmax)

    def call(self, x, is_training=False):
        x = self.input_layer(x)

        x = self.dense1(x)
        x = self.dense2(x)

        if not is_training:
            x = self.softmax(x)

        return x

model = MyModel()

# train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

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
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from layers.BitInput import BitInput
from layers.BitLayer import BitLayer
from layers.BitOutput import BitOutput


# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# create model
class MyModel(Model):

    def __init__(self, N):
        super(MyModel, self).__init__()

        self.input_layer = layers.Flatten()
        self.bit_creator = BitInput(N)

        self.dense1 = BitLayer(128, N)
        self.dense2 = BitLayer(128, N)

        self.dense3 = BitLayer(10, N)
        self.bit_destroyer = BitOutput()

    def call(self, x, is_training=False):
        x = self.input_layer(x)
        x = self.bit_creator(x)

        x = self.dense1(x)
        x = self.dense2(x)

        x = self.dense3(x)
        x = self.bit_destroyer(x)
        

        return x

model = MyModel(256)


# train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

model.summary()

# evaluate model
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss: {}\nAccuracy: {}".format(val_loss, val_acc))


# evaluate prediction
predictions = model.predict(x_test)


print("Prediction: {}".format(np.argmax(predictions[0])))
plt.imshow(x_test[0], cmap=plt.cm.binary)
#plt.savefig('main.png')
plt.show()

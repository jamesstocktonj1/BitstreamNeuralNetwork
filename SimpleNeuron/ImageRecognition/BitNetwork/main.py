import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from BitModel import BitModel


# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# reshape from (len, 28, 28) to (len, 28 * 28)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# results arrays
data_range = 25
thresholds = list(t / data_range for t in range(data_range))

loss_data = []
accuracy_data = []


# threshold test array
for t in thresholds:

    print("\nTesting with Threshold {}".format(t))

    # round any values greater than 0.0 to be equal to 1.0 (bit input)
    x_temp_train = np.ceil(x_train - t)
    x_temp_test = np.ceil(x_test - t)

    # create model
    model = BitModel()

    # train model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_temp_train, y_train, epochs=3)

    # evaluate model
    val_loss, val_acc = model.evaluate(x_temp_test, y_test)
    print("Loss: {}\nAccuracy: {}".format(val_loss, val_acc))

    loss_data.append(val_loss)
    accuracy_data.append(val_acc)

    # save image
    plt.imshow(x_temp_test[0].reshape(28, 28), cmap=plt.cm.binary)
    plt.savefig("data/threshold_{}.png".format(t))


# plot data / threshold graphs
fig1 = plt.subplot(2, 1, 1)
fig1.set_title("Loss / Threshold")
fig1.plot(loss_data)

fig2 = plt.subplot(2, 1, 2)
fig2.set_title("Accuracy / Threshold")
fig2.plot(accuracy_data)

plt.title("Threshold Analysis")

plt.savefig("data/threshold_analysis.png")
#plt.show()
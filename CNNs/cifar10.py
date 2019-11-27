import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Setting class names for the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
plt.imshow(X_test[10])
plt.show()

model = tf.keras.models.Sequential()

# Add first CNN layer
# Note on padding: size of output is same as input
# Shape is 32x32 and 3 color channels
model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

# Add second CNN layer and max pool
# Pooling helps computational cost and prevents overfitting
model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Add third layer
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=3, padding="same", activation="relu"))

# Add the fourth layer and max pool
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

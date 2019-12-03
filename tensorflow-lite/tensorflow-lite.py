import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape dataset
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(
    units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])

model.fit(X_train, y_train, epochs=5)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy is {}".format(test_accuracy))

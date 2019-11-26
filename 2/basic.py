import numpy as py
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Since each image is 28x28, we simply use reshape the full dataset to [-1 (all elements), height * width]
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

###
# Creating the network
###

# Model
model = tf.keras.models.Sequential()

# Add the first layer
# 128 neurons, ReLU activation fn, input shape is (784)
model.add(tf.keras.layers.Dense(
    units=128, activation='relu', input_shape=(784, )))

# Add a dropout layer - randomly drop neurons so we don't overfit
model.add(tf.keras.layers.Dropout(0.2))

# Add a second layer - output layer
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compiling model - Adam optimizer, sparse softmax crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
              
model.summary()

###
# Using the model
###

model.fit(X_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}\nTest loss: {}".format(test_accuracy, test_loss))

import numpy as py
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

#Since each image is 28x28, we simply use reshape the full dataset to [-1 (all elements), height * width]
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)


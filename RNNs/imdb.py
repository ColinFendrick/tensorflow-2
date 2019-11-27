import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Dataset parameters
number_of_words = 20000
max_len = 100

# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# Padd all sequences to be the same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=max_len)

X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=max_len)

# Embed layer parameters
vocab_size = number_of_words
embed_size = 128

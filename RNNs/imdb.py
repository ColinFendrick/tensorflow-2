import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Dataset parameters
number_of_words = 20000
max_len = 100

# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# Pad all sequences to be the same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=max_len)

X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=max_len)

# Embed layer parameters
vocab_size = number_of_words
embed_size = 128

model = tf.keras.models.Sequential()

# Adding the embeding layer
model.add(tf.keras.layers.Embedding(
    vocab_size, embed_size, input_shape=(X_train.shape[1],)))

# Add the LSTM Layer
# 128 units, tanh activation - usual activation for lstm
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.2))

# Adding the dense output layer - only 1 output neuron since it is a binary choice
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile, fit, evaluate
# rmsprop is recommended for rnn
# binary_crossentropy is because we hace a binary choice model
# accuracy metric is used in binary models
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=128)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
model.summary()
print("Test accuracy: {}\nTest loss: {}".format(test_accuracy, test_loss))

import tensorflow as tf
import numpy as np

# Constant
tensor_20 = tf.constant([[23, 4], [32, 51]])
# Shape of constant
print(tensor_20.shape, '\n')

# Getting values straight from a TensorFlow constant to numpy - withut session - new in 2.0
print(tensor_20.numpy(), '\n')

# We are able to convert a numpy matrix back to a TensorFlow tensor as well
print('numpy matrix to tf: ', tf.constant(np.array([[12, 7], [19, 1]])), '\n')

# Operations
tensor = tf.constant([[1, 2], [3, 4]])
print('[[1,2],[3,4]] + 2 = ', tensor+2, '\n')
print('[[1,2],[3,4]] * 5 = ', tensor*5, '\n')

# Using numpy operations
print('squared tensor = ', np.square(tensor), '\n')
print('sqrt =', np.sqrt(tensor), '\n')

# Dot product
print('[[1,2],[3,4]]\u00B7[[23,4],[32,51]] =' , np.dot(tensor, tensor_20))

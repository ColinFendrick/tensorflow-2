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

# Operations with variables
tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(tf2_variable, '\n')
print('variable.numpy = {}'.format(tf2_variable.numpy()), '\n')

# Strings
tf_string = tf.constant('Tensorflow string')
print('tf string length = ', tf.strings.length(tf_string), '\n')
print('tf string unicode decode = ', tf.strings.unicode_decode(tf_string, 'UTF8'), '\n')
tf_string_array = ['an', 'array', 'of', 'strings']
for i, s in enumerate(tf_string_array):
    print('{} in the array is = {}\n'.format(i, s))


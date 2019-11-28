import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "./transfer-learning/cats_and_dogs_filtered.zip"
zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall("./")
zip_object.close()
dataset_path_new = "./cats_and_dogs_filtered/"
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

# Loading the pre-trained model
IMG_SHAPE = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

# base_model.summary()

# Freeze the base model
base_model.trainable = False

# Define a custom head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(
    units=1, activation='sigmoid')(global_average_layer)

# Define the model
model = tf.keras.models.Model(
    inputs=base_model.input, outputs=prediction_layer)
# model.summary()

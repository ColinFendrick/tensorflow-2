from __future__ import print_function
import tempfile
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema

# Load data
dataset = pd.read_csv("data-validation/pollution-small.csv")

# Drop the data column
features = dataset.drop("Date", axis=1)  # Don't need the date column
print(features.head())  # Show the head columns

# Tf-transform requires dictionaries, convert to a list of dicts
dict_features = list(features.to_dict("index").values())

# Define the dataset metadata
data_metadata = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec({
        "no2": tf.io.FixedLenFeature([], tf.float32),
        "so2": tf.io.FixedLenFeature([], tf.float32),
        "pm10": tf.io.FixedLenFeature([], tf.float32),
        "soot": tf.io.FixedLenFeature([], tf.float32),
    })
)


# Preprocessing function
def preprocessing_fn(inputs):
    no2 = inputs['no2']
    pm10 = inputs['pm10']
    so2 = inputs['so2']
    soot = inputs['soot']

    no2_normalized = no2 - tft.mean(no2)
    so2_normalized = so2 - tft.mean(so2)

    pm10_normalized = tft.scale_to_0_1(pm10)
    soot_normalized = tft.scale_by_min_max(soot)

    return {
        "no2_normalized": no2_normalized,
        "so2_normalized": so2_normalized,
        "pm10_normalized": pm10_normalized,
        "soot_normalized": soot_normalized
    }


# Tensorflow Transform uses Apache Beam in the background to perform scalable data transforms. In this function we will use a direct runner.
# Arguments to provide to the runner:
# dict_features - This is our dataset converted into Python Dictionary.
# data_metadata - This is our mada data for the dataset that we have created.
# preprocessing_fn - The main preprocessing function. Called to perform preprocessing operation per column.

# This is a special syntax used in Apache Beam. This is used to stack operations and invoke transforms on our data.
# result = data_to_pass | where_to_pass_the_data

# Our use case:
# result -> transformed_dataset, transform_fn
# data_to_pass -> (dict_features, data_metadata)
# where_to_pass_the_data -> tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
# transformed_dataset, transform_fn = ((dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

# Data transform
def data_transform():

    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = (
            (dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

    transformed_data, transformed_metadata = transformed_dataset

    for i in range(len(transformed_data)):
        print("Raw: ", dict_features[i])
        print("Transformed:", transformed_data[i])


data_transform()

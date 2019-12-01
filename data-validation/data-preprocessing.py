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
features = dataset.drop("Date", axis=1)
print(features.head())

# Define the dataset metadata
data_metadata = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec({
        "no2": tf.FixedLenFeature([], tf.float32),
        "so2": tf.FixedLenFeature([], tf.float32),
        "pm10": tf.FixedLenFeature([], tf.float32),
        "soot": tf.FixedLenFeature([], tf.float32),
    }
    )
)

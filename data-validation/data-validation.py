from __future__ import print_function
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv

# Read the dataset
dataset = pd.read_csv("data-validation/pollution-small.csv")

training_data = dataset[:1600]
training_data.describe()

test_set = dataset[1600:]
test_set.describe()

# Generate training data statistics
train_stats = tfdv.generate_statistics_from_dataframe(dataframe=dataset)
# Infer schema
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)

# Generate test statistics
test_stats = tfdv.generate_statistics_from_dataframe(dataframe=test_set)

# To compare test statistics with schema, check for anomalies
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
tfdv.display_anomalies(anomalies)

# Create new data with anomalies
test_set_copy = test_set.copy()
test_set_copy.drop("soot", axis=1, inplace=True)

# Show statistics based on data with anomalies
test_set_copy_stats = tfdv.generate_statistics_from_dataframe(
    dataframe=test_set_copy)
anomalies_new = tfdv.validate_statistics(
    statistics=test_set_copy_stats, schema=schema)
tfdv.display_anomalies(anomalies_new)

# Prepare schema for serving
schema.default_environment.append("TRAINING")
schema.default_environment.append("SERVING")
# Remove target column from serving schema
tfdv.get_feature(schema, "soot").not_in_environment.append("SERVING")
# Check for anomalies between serving environment and new test set
serving_env_anomalies = tfdv.validate_statistics(
    test_set_copy_stats, schema, environment="SERVING")
tfdv.display_anomalies(serving_env_anomalies)

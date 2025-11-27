"""
Dummy Dataset Generator

This script generates a synthetic dataset for testing and demonstrating
data preprocessing and machine learning pipelines. The dataset includes
numeric features, categorical variables, missing values, and outliers
to simulate real-world data quality issues.

Dataset Characteristics
-----------------------
- **Feature1**: Numeric feature with normal distribution (mean=100, std=10),
  includes missing values and outliers
- **Feature2**: Integer feature with uniform distribution (range: 0-99)
- **Category**: Categorical feature with 4 classes ('A', 'B', 'C', 'D'),
  includes missing values
- **Target**: Binary classification target (0 or 1)

Size
----
102 rows × 4 columns

Output
------
Creates 'dummy_dataset.csv' in the current directory

Usage
-----
Run this script to generate a test dataset:
    $ python generate_and_load_dummy_data.py

Example
-------
>>> # After running, load the generated dataset
>>> df = pd.read_csv('dummy_dataset.csv')
>>> print(df.shape)
(102, 4)
"""

import pandas as pd
import numpy as np

# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    # Normally distributed with an outlier and missing value
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],
    # Random integers between 0 and 99
    'Feature2': np.random.randint(0, 100, 102).tolist(),
    # Categorical variable with some missing values
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],
    # Binary target variable
    'Target': np.random.choice([0, 1], 102).tolist()
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Display the first few rows of the dummy dataset
print(df_dummy.head())

# Save the dummy dataset to a CSV file
df_dummy.to_csv('Data cleaning and preprocessing/dummy_dataset.csv', index=False)

#!/usr/bin/env python3
# Identifying and addressing errors in data collection is a critical step in preparing
# high-quality datasets for analysis and ML. This walkthrough provided a comprehensive
# guide to detecting and correcting common data collection errors, including missing 
# values, outliers, data entry errors, and inconsistencies.
#
# By applying these techniques, you can significantly improve the reliability and accuracy 
# of your data, leading to better outcomes in your AI and ML projects. As you continue 
# to work with data, these skills will help you maintain the integrity of your datasets, 
# ensuring that your models are built on a solid foundation of clean, accurate data.
# 
# Describe the structure and characteristics of a dataset.
# Identify and handle missing values effectively.
# Detect and manage outliers using statistical methods.
# Identify and correct data entry errors for consistency.
# Validate data consistency and ensure high-quality datasets.

from xml.parsers.expat import errors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load and inspect dataset
df = pd.read_csv('dummy_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the data types of each column
print(df.dtypes)

# Step 2: Identify missing values
# Check for missing values in the dataset
missing_values = df.isnull().sum()

# Display columns with missing values
print(missing_values[missing_values > 0])

#Handle missing values
#Once identified, missing values can be handled through several methods:
# A 
# Remove rows/columns:
df_cleaned_a = df.dropna()  # Remove rows with any missing values

# B
# Fill missing values:
df_cleaned_b = df.fillna(df.mean())  # Fills missing numeric values with the mean of the column

# Step 3: Detect outliers using statistical methods
# 
# Use descriptive statistics and visualization to detect outliers
# Descriptive statistics such as the mean and standard deviation, 
# along with visual tools such as box plots, help to identify outliers. 
# Z-scores quantify how far a data point is from the mean, 
# with a Z-score greater than 3 typically considered an outlier.

# Use descriptive statistics to identify potential outliers
print(df.describe())

# Visualize data to spot outliers using box plots
df.boxplot(column=['Column1', 'Column2'])  # Replace with actual column names
plt.show()

# Calculate Z-scores to identify outliers
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

# Find rows with Z-scores greater than 3
outliers = (z_scores > 3).all(axis=1)
print(df[outliers])

# Handle outliers
# Outliers can be handled by either removing or transforming them:

# A. Remove outliers:
df_no_outliers = df[(z_scores < 3).all(axis=1)]

# B. Transform outliers:
df['Column1'] = np.where(df['Column1'] > upper_limit, upper_limit, df['Column1'])  # Replace with actual column name and upper_limit
# Depending on the situation, you may choose to remove outliers to prevent them 
# from skewing your results or transform them to minimize their impact.

# Step 4: Check for data entry errors
# Data entry errors, such as incorrect values or inconsistent formatting, can be subtle but impactful.
# Check for unique values in categorical columns to identify inconsistencies
print(df['CategoryColumn'].unique())  # Replace with actual column name

# Use value counts to identify unusual or erroneous entries
print(df['CategoryColumn'].value_counts())

# Check numeric columns for impossible values (e.g., negative ages)
print(df[df['Age'] < 0])  # Replace “Age” with the actual column name

# By examining the unique values and frequency distributions in categorical columns, 
# you can identify inconsistencies, such as misspellings or unexpected categories. 
# For numeric data, you can look for impossible or implausible values that may 
# indicate data entry errors.

# Correct data entry errors
# After identifying these errors, you can correct them as follows:
# A. Standardize categories:
df['CategoryColumn'] = df['CategoryColumn'].str.strip().str.lower().replace({'misspelled': 'correct'})  # Example replacement

# B. Correct numeric errors:
df['Age'] = np.where(df['Age'] < 0, np.nan, df['Age'])  # Replace negative ages with NaN

# Standardizing categories ensures consistency in your data, while correcting numeric 
# errors prevents them from negatively impacting your analysis.

# Step 5: Validate data consistency
# Consistency checks help to ensure that your data behaves as expected over time or across
# different variables.

# Perform consistency checks
# Cross-validate data consistency between related columns
df['Total'] = df['Part1'] + df['Part2']  # Replace with actual column names
inconsistent_rows = df[df['Total'] != df['ExpectedTotal']]  # Replace with the actual column for the expected total
print(inconsistent_rows)

# Check for duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)

# Address inconsistencies
# Correct inconsistencies in the following ways:
#
# A. Correct calculation errors:
df['ExpectedTotal'] = df['Part1'] + df['Part2']  # Recalculate totals if they were incorrectly entered

# B. Remove duplicates:
df_no_duplicates = df.drop_duplicates()

# By recalculating values and removing duplicates, you ensure that your dataset is consistent and free from redundancy.

"""
Data Cleaning and Preprocessing Module

This module provides a comprehensive suite of functions for preparing raw data
for machine learning and statistical analysis. It handles common data preprocessing
tasks including missing value imputation, outlier detection and removal, feature
scaling, and categorical variable encoding.

Key Features
------------
- **Data Loading**: Load datasets from CSV files
- **Missing Value Handling**: Impute missing values using mean substitution
- **Outlier Detection**: Remove outliers using Z-score method (statistical approach)
- **Feature Scaling**: Standardize numeric features using Z-score normalization
- **Categorical Encoding**: Convert categorical variables to numeric format using one-hot encoding
- **Data Export**: Save processed datasets to CSV files

Dependencies
------------
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scipy: Statistical functions (Z-score calculation)
- scikit-learn: Machine learning preprocessing tools (StandardScaler, MinMaxScaler)
- missingno: Optional visualization of missing data patterns

Typical Workflow
----------------
1. Load raw data from CSV file
2. Handle missing values in numeric columns
3. Remove statistical outliers
4. Scale numeric features for model compatibility
5. Encode categorical variables
6. Save the cleaned and preprocessed dataset

Example
-------
>>> # Complete preprocessing pipeline
>>> df = load_data('raw_data.csv')
>>> df = handle_missing_values(df)
>>> df = remove_outliers(df)
>>> df = scale_data(df)
>>> df = encode_categorical(df, ['category_col1', 'category_col2'])
>>> save_data(df, 'preprocessed_data.csv')

Notes
-----
- All functions assume the input DataFrame contains primarily numeric data
- Categorical encoding should be performed after scaling numeric features
- Consider the data distribution when choosing preprocessing methods
- Always validate the preprocessed data before using it for modeling

Version
-------
1.0.0
"""
import pandas as pd
import numpy as np
from scipy import stats  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
#from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore
#import missingno as msno  # Optional: for visualizing missing data

def load_data(filepath):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file to be loaded.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    
    Examples
    --------
    >>> df = load_data('data/my_dataset.csv')
    """
    return pd.read_csv(filepath)

def handle_missing_values(dataframe):
    """
    Handle missing values in numeric columns by filling them with the mean.
    
    This function fills NaN values in each numeric column with that column's mean.
    Non-numeric columns are not affected.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the data with potential missing values.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled with column means.
    
    Notes
    -----
    This method only works well for numeric data. For categorical data,
    consider using mode or a different imputation strategy.
    
    Examples
    --------
    >>> df_cleaned = handle_missing_values(df)
    """
    return dataframe.fillna(dataframe.mean())

def remove_outliers(dataframe):
    """
    Remove outliers from the dataset using the Z-score method.
    
    Rows containing values with a Z-score greater than 3 (in absolute value)
    are considered outliers and removed from the dataset.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing numeric data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with outlier rows removed.
    
    Notes
    -----
    - This method assumes data is approximately normally distributed.
    - A Z-score threshold of 3 means values beyond 3 standard deviations
      from the mean are considered outliers.
    - Only works with numeric columns.
    
    Examples
    --------
    >>> df_no_outliers = remove_outliers(df)
    """
    z_scores = np.abs(np.array(stats.zscore(dataframe)))
    return dataframe[(z_scores < 3).all(axis=1)]

def scale_data(dataframe):
    """
    Scale numeric data using standardization (Z-score normalization).
    
    Transforms features by removing the mean and scaling to unit variance.
    The standard score of a sample x is calculated as: z = (x - u) / s
    where u is the mean and s is the standard deviation.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing numeric data to be scaled.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with scaled values (mean=0, std=1).
    
    Notes
    -----
    StandardScaler is sensitive to outliers. Consider removing outliers
    before scaling, or use RobustScaler for data with outliers.
    
    Examples
    --------
    >>> df_scaled = scale_data(df)
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)

def encode_categorical(dataframe, categorical_columns):
    """
    Encode categorical variables using one-hot encoding.
    
    Converts categorical variables into a series of binary (0/1) columns,
    one for each category. This is also known as dummy variable encoding.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the data.
    categorical_columns : list of str
        List of column names to be one-hot encoded.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns converted to one-hot encoded columns.
        Original categorical columns are replaced with multiple binary columns.
    
    Examples
    --------
    >>> df_encoded = encode_categorical(df, ['color', 'size'])
    
    Notes
    -----
    For columns with many categories, consider using ordinal encoding
    or target encoding to avoid creating too many columns.
    """
    return pd.get_dummies(dataframe, columns=categorical_columns)

def save_data(dataframe, output_filepath):
    """
    Save DataFrame to a CSV file.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to be saved.
    output_filepath : str
        Path where the CSV file will be saved.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> save_data(df, 'output/processed_data.csv')
    
    Notes
    -----
    The index is not saved to the CSV file (index=False).
    """
    dataframe.to_csv(output_filepath, index=False)

# Example usage:
df = load_data('your_dataset.csv')
df = handle_missing_values(df)
df = remove_outliers(df)
df = scale_data(df)
df = encode_categorical(df, ['categorical_column_name'])
save_data(df, 'cleaned_preprocessed_data.csv')

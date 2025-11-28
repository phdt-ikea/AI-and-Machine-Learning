"""Secure machine learning pipeline: validates input data, trains a logistic regression model,
encrypts and saves it, then verifies integrity on load before evaluation.

Raises:
    ValueError: If input data contains null values.
    ValueError: If the loaded model fails integrity verification.

Returns:
    None
"""

import os
import pickle
import hashlib
from cryptography.fernet import Fernet # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

# Validate and sanitize input data
def validate_data(df):
    """Validate and sanitize the input dataset.

    Parameters:
        df (pandas.DataFrame): The dataset to validate.

    Raises:
        ValueError: If the dataset contains null values.

    Returns:
        pandas.DataFrame: The validated dataset.
    """
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values. Please clean the data before processing.")
    # Additional validation checks can be added here
    return df

# Load and validate dataset
data = validate_data(pd.read_csv('user_data.csv'))

# Split the dataset into features and target with validation
X = validate_data(data.iloc[:, :-1])
y = validate_data(data.iloc[:, -1])

# Split the data into training and testing sets with a securely managed random state
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=int.from_bytes(os.urandom(16), "little"))

# Using the earlier securely managed split; removing redundant fixed random_state split
# Train a simple logistic regression model (Flaw: No model security checks)
model = LogisticRegression()
model.fit(X_train, y_train)

# Encrypt model before saving
key = Fernet.generate_key()
cipher = Fernet(key)

# Save the encrypted model to disk
FILENAME = 'finalized_model.sav'
encrypted_model = cipher.encrypt(pickle.dumps(model))
with open(FILENAME, 'wb') as f:
    f.write(encrypted_model)

# Load the encrypted model from disk and verify its integrity
with open(FILENAME, 'rb') as fr:
    encrypted_model = fr.read()
    decrypted_model = cipher.decrypt(encrypted_model)

loaded_model = pickle.loads(decrypted_model)

# Compute hash of the loaded model
LOADED_MODEL_HASH = hashlib.sha256(decrypted_model).hexdigest()

# Verify that the loaded model's hash matches the original
ORIGINAL_MODEL_HASH = hashlib.sha256(pickle.dumps(model)).hexdigest()
if LOADED_MODEL_HASH != ORIGINAL_MODEL_HASH:
    raise ValueError("Model integrity check failed. The model may have been tampered with.")

result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')

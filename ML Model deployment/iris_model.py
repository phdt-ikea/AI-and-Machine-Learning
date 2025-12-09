"""
Iris Classification Model Training and Deployment

This script trains a logistic regression model on the Iris dataset and saves it
for deployment. The Iris dataset is a classic multiclass classification problem
containing measurements of iris flowers from three different species.

Dataset
-------
The Iris dataset consists of 150 samples with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Target classes (3 species):
- Setosa (0)
- Versicolor (1)
- Virginica (2)

Model
-----
- Algorithm: Logistic Regression (multiclass)
- Maximum iterations: 200
- Solver: Default (lbfgs)

Training Configuration
----------------------
- Train/Test split: 80/20
- Random state: 42 (for reproducibility)
- Training samples: 120
- Test samples: 30

Output
------
Saves trained model to './iris_model.pkl' using joblib serialization.

Usage
-----
Train the model and save it:
    $ python iris_model.py

Load the saved model:
    >>> import joblib
    >>> model = joblib.load('./iris_model.pkl')
    >>> predictions = model.predict(new_data)

Dependencies
------------
- scikit-learn: Machine learning library for model and dataset
- joblib: Efficient serialization of Python objects

Notes
-----
- The model file can be loaded and used for making predictions on new iris samples
- Model achieves high accuracy (>95%) on this well-separated dataset
- Suitable for demonstrating basic ML model deployment workflows

Example Prediction
------------------
>>> import joblib
>>> import numpy as np
>>> model = joblib.load('./iris_model.pkl')
>>> # Predict for a sample: [sepal_length, sepal_width, petal_length, petal_width]
>>> sample = np.array([[5.1, 3.5, 1.4, 0.2]])
>>> prediction = model.predict(sample)
>>> print(f"Predicted class: {prediction[0]}")  # Expected: 0 (Setosa)
"""
from sklearn.datasets import load_iris # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
import joblib # type: ignore

# Load the dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, './iris_model.pkl')

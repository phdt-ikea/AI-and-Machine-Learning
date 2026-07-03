"""
Iris Model REST API Server

This Flask application serves the trained Iris classification model via a REST API.
It provides a prediction endpoint that accepts iris flower measurements and returns
the predicted species class.

Endpoints
---------
POST /predict
    Accepts iris flower measurements and returns species prediction.

API Usage
---------
Request Format:
    POST /predict
    Content-Type: application/json
    
    Body:
    {
        "input": [sepal_length, sepal_width, petal_length, petal_width]
    }

Response Format:
    {
        "prediction": <class_id>
    }
    
    Where class_id is:
    - 0: Iris Setosa
    - 1: Iris Versicolor
    - 2: Iris Virginica

Examples
--------
Using curl:
    $ curl -X POST http://localhost:80/predict \\
        -H "Content-Type: application/json" \\
        -d '{"input": [5.1, 3.5, 1.4, 0.2]}'
    
    Response: {"prediction": 0}

Using Python requests:
    >>> import requests
    >>> url = 'http://localhost:80/predict'
    >>> data = {'input': [5.1, 3.5, 1.4, 0.2]}
    >>> response = requests.post(url, json=data)
    >>> print(response.json())
    {'prediction': 0}

Server Configuration
--------------------
- Host: 0.0.0.0 (accessible from all network interfaces)
- Port: 80 (default HTTP port)
- Model: iris_model.pkl (must exist in same directory)

Dependencies
------------
- flask: Web framework for API endpoints
- joblib: Model deserialization
- numpy: Array manipulation for input data

Usage
-----
Start the server:
    $ python serve_model.py

Or with custom port:
    $ python serve_model.py

Notes
-----
- Ensure iris_model.pkl exists before starting the server
- Port 80 requires root/admin privileges on most systems
- For production, consider using a WSGI server (gunicorn, uWSGI)
- Add error handling and input validation for production use
- The model expects exactly 4 numerical features in the input array

Security Considerations
-----------------------
- This is a basic implementation for demonstration purposes
- Add authentication/authorization for production deployments
- Implement rate limiting to prevent abuse
- Add input validation and sanitization
- Use HTTPS for encrypted communication
"""

from flask import Flask, request, jsonify
import joblib # type: ignore
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict iris species from flower measurements.
    
    Accepts POST requests with iris flower measurements and returns the
    predicted species class using the pre-trained logistic regression model.
    
    Request Body
    ------------
    JSON object with:
        input : list of float
            Array of 4 measurements: [sepal_length, sepal_width, 
            petal_length, petal_width] in centimeters.
    
    Returns
    -------
    JSON response:
        prediction : int
            Predicted class (0=Setosa, 1=Versicolor, 2=Virginica).
    
    Examples
    --------
    Request:
        POST /predict
        {"input": [5.1, 3.5, 1.4, 0.2]}
    
    Response:
        {"prediction": 0}
    
    Notes
    -----
    - Input array is reshaped to (1, 4) for single sample prediction
    - Prediction is converted to int for JSON serialization
    - force=True allows parsing even without Content-Type header
    
    Raises
    ------
    - KeyError: If 'input' key is missing from request body
    - ValueError: If input cannot be converted to numpy array
    - Exception: If model prediction fails
    """
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

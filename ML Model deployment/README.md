# Preparing a model for deployment
## Introduction
ML Model deployment contains iris_model.py ML model to exrecise model deployment with docker. This exercise will give you hands-on experience with the steps involved in packaging, containerizing, and deploying a model in a production environment.

## Objective
* Package an ML model along with its dependencies.
* Containerize the model using Docker.
* Deploy the model locally, and test its performance.

## Prerequisites
* Python 3.14: for running scripts and packaging the model
* Docker: for containerizing the model
* An IDE: such as VS Code, PyCharm, or Jupyter Notebook

## Steps
1. Packaging the model - Running command python iris_model.py will save model iris_model.pkl
2. Python script for serving the model - serve_model.py
3. Containerizing the model with Docker - docker build -t iris_model_image .
4. Run the Docker container - docker run -d -p 80:80 iris_model_image
5. Test the deployment - curl -X POST http://127.0.0.1:80/predict -H "Content-Type: application/json" -d '{"input": [5.1, 3.5, 1.4, 0.2]}'
Will give response: {"prediction":0}
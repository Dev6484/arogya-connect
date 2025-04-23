# ML Model for Disease Prediction (Team Use)

This folder contains the trained machine learning models and utilities for predicting diseases based on symptoms. It is designed to be integrated into the backend of our web app.


## 📦 Setup

1. Install dependencies:

pip install -r requirements.txt


from src.utils import predict_disease

# Input should be list of symptoms from UI, example:
input_symptoms = ["fever", "headache", "nausea"]
prediction = predict_disease(input_symptoms)
print(prediction)

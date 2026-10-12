# Diabetes Prediction Model
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = 'diabetes_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

scaler = StandardScaler()
# Fit scaler with typical diabetes dataset statistics
# (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
scaler.fit(np.array([
    [3.8, 120.9, 69.1, 20.5, 79.8, 31.9, 0.47, 33.2]  # Mean values
]))
scaler.scale_ = np.array([3.37, 31.97, 19.36, 16.07, 115.24, 7.88, 0.33, 11.76])  # Std dev values

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    preg = float(request.json['pregnancies'])
    glucose = float(request.json['glucose'])
    bp = float(request.json['blood_pressure'])
    skin = float(request.json['skin_thickness'])
    insulin = float(request.json['insulin'])
    bmi = float(request.json['bmi'])
    dpf = float(request.json['diabetes_pedigree_function'])
    age = float(request.json['age'])

    # Create feature array and normalize
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)

    # Make a prediction
    prediction = model.predict(features_scaled)

    # Return the result
    if prediction[0] == 1:
        return jsonify({'prediction': 1})
    else:
        return jsonify({'prediction': 0})

if __name__ == '__main__':
    app.run(debug=True)
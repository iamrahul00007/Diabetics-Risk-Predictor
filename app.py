import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

scaler = StandardScaler()
scaler.fit(np.array([
    [3.8, 120.9, 69.1, 20.5, 79.8, 31.9, 0.47, 33.2]  # Mean values
]))
scaler.scale_ = np.array([3.37, 31.97, 19.36, 16.07, 115.24, 7.88, 0.33, 11.76])  # Std dev values

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetes_pedigree_function']),
            float(data['age']),
        ]])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid or missing input'}), 400

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)

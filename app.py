"""
Assignment 3: User Interface for ML Models
Backend API for Absenteeism Prediction Model
"""

import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------- #
#  Flask configuration
# ---------------------------------------------------------------------------- #
app = Flask(__name__, static_folder='frontend/dist', static_url_path='')

CORS(app, origins=[
    'http://localhost:5173',
    'http://localhost:3000',
    'http://localhost:5000',
    'https://assignment4-hyu2.onrender.com',  # ✅ your Render frontend
    'https://assignment-4-o9gt.onrender.com',
    'https://assig4-7h6s.onrender.com'
])

# ---------------------------------------------------------------------------- #
#  Global variables
# ---------------------------------------------------------------------------- #
model = None
scaler = None
feature_columns = None


# ---------------------------------------------------------------------------- #
#  Load model safely (works locally & on Render)
# ---------------------------------------------------------------------------- #
def load_model():
    """Load the trained model and scaler from file"""
    global model, scaler, feature_columns

    try:
        # Absolute, platform-safe path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'model.pkl')

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_names']

        print("✅ Model loaded successfully")

    except FileNotFoundError:
        print("❌ Model file not found. Please ensure 'model.pkl' is in the same folder as app.py.")
        model, scaler, feature_columns = None, None, None


# ---------------------------------------------------------------------------- #
#  Health check route
# ---------------------------------------------------------------------------- #
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'feature_columns': len(feature_columns) if feature_columns is not None else 0
    })


# ---------------------------------------------------------------------------- #
#  Preprocess user input
# ---------------------------------------------------------------------------- #
def preprocess_input(data):
    """Convert JSON input into the format expected by the model"""
    df = pd.DataFrame([data])

    categorical_cols = [
        'Reason for absence', 'Month of absence', 'Day of the week', 'Seasons',
        'Hit target', 'Disciplinary failure', 'Education', 'Son',
        'Social drinker', 'Social smoker', 'Pet'
    ]

    available_cat = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=available_cat, drop_first=True)

    # Add missing columns
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    return df_encoded


# ---------------------------------------------------------------------------- #
#  Serve frontend
# ---------------------------------------------------------------------------- #
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_react_app(path):
    return send_from_directory(app.static_folder, 'index.html')


# ---------------------------------------------------------------------------- #
#  Prediction API
# ---------------------------------------------------------------------------- #
@app.route('/api/predict', methods=['POST'])
def predict():
    """Return absenteeism prediction from model"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded on server'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        processed_data = preprocess_input(data)
        scaled_data = scaler.transform(processed_data)
        prediction = model.predict(scaled_data)[0]

        return jsonify({
            'prediction': float(prediction),
            'confidence': 0.8,
            'message': f'Predicted absenteeism: {prediction:.2f} hours'
        })

    except Exception as e:
        import traceback
        print("❌ Error during prediction:\n", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------- #
#  Model info & feature importance routes
# ---------------------------------------------------------------------------- #
@app.route('/api/model_info', methods=['GET'])
def model_info():
    try:
        info = {
            'model_type': 'Linear Regression',
            'performance': {
                'baseline': {'rmse': 11.4292, 'mae': 6.4389, 'r2_score': -0.1987},
                'mitigated': {'rmse': 43.1228, 'mae': 16.5046, 'r2_score': -0.0875}
            },
            'fairness_metrics': {
                'baseline': {
                    'age_group_mae_gap': 20.78,
                    'education_mae_gap': 13.36,
                    'service_time_mae_gap': 3.76
                },
                'mitigated': {
                    'age_group_mae_gap': 0.00,
                    'education_mae_gap': 17.56,
                    'service_time_mae_gap': 0.00
                }
            },
            'bias_mitigation': {
                'applied': True,
                'measures': [
                    'Removed proxy features (Height, Weight, BMI)',
                    'Balanced age group representation',
                    'Balanced education level representation'
                ]
            },
            'limitations': [
                'Model performance limited due to data imbalance',
                'Predictions may not be accurate for extreme cases',
                'Model trained on specific industry dataset; may not generalize'
            ]
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    try:
        if model is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 500

        importance = [
            {'feature': feature, 'importance': abs(model.coef_[i])}
            for i, feature in enumerate(feature_columns)
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)
        return jsonify(importance[:10])

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------- #
#  Main entry point
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    load_model()
    # host 0.0.0.0 required for Render
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
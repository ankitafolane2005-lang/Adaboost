import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = "adaboost.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Please place it in the same directory.")

# Expected feature names extracted from the model
FEATURE_NAMES = [
    "age", "gender", "state", "family_income", "high_school_gpa",
    "sat_score", "act_score", "attendance_rate", "ap_courses",
    "extracurricular_count", "volunteer_hours", "leadership_positions",
    "coding_projects", "social_media_hours", "online_certifications",
    "essay_score", "recommendation_score", "interview_score"
]

# Simple HTML form UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Model Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
        .container { max-width: 600px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"], input[type="number"] { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { width: 100%; padding: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 4px; font-weight: bold; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h2>AdaBoost Prediction Form</h2>
        <form action="/predict_ui" method="post">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature.replace('_', ' ').title() }}:</label>
                <input type="number" step="any" name="{{ feature }}" id="{{ feature }}" required>
            </div>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            Prediction Output: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None)

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    if model is None:
        return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction="Error: Model file not loaded.")
    
    try:
        # Extract features from form input
        features = [float(request.form.get(feature, 0)) for feature in FEATURE_NAMES]
        features_array = np.array([features])
        
        # Predict
        prediction = model.predict(features_array)[0]
        return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=str(prediction))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=f"Error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON API Endpoint.
    Expects JSON payload with key 'features' (list of 18 floats) or dictionary with feature names.
    """
    if model is None:
        return jsonify({'error': 'Model file not loaded.'}), 500

    try:
        data = request.get_json()
        
        if 'features' in data:
            input_data = np.array(data['features']).reshape(1, -1)
        else:
            # Handle dictionary mapping key feature names
            input_data = np.array([[float(data.get(feat, 0)) for feat in FEATURE_NAMES]])
            
        prediction = model.predict(input_data)
        
        response = {
            'prediction': int(prediction[0]) if hasattr(prediction[0], 'item') else str(prediction[0])
        }
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0].tolist()
            response['probabilities'] = probabilities

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
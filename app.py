import os
import logging
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Skip ngrok warning
@app.before_request
def skip_ngrok_warning():
    if request.headers.get("ngrok-skip-browser-warning") != "true":
        return jsonify({"error": "ngrok-skip-browser-warning header is missing"}), 403

# Disable GPU and suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Model path
model_path = '/content/drive/MyDrive/Backend/final_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_path}")

# Load the trained model
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"🚫 Failed to load model: {str(e)}")

# Function to preprocess CT scan image
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"🚫 Image preprocessing failed: {str(e)}")

# Route for COVID-19 prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'ct_scan' not in request.files:
        return jsonify({'error': '🚫 No CT scan uploaded'}), 400

    file = request.files['ct_scan']
    image_path = 'temp_ct_scan.png'
    file.save(image_path)

    try:
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        covid_probability = prediction[0][0] * 100

        result = {'COVID-19 Likelihood': f"{covid_probability:.2f}%"}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

import os
import logging
import gdown
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename  # Fix file name issues

app = Flask(__name__)

# Disable GPU and suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)

# Google Drive file ID
GDRIVE_FILE_ID = "1F3CyYozjJlPSfKanlm0n2SmqOyT5rQ8L"
MODEL_PATH = "final_model.h5"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("📥 Downloading model from Google Drive...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        except Exception as e:
            logging.error(f"🚫 Model download failed: {str(e)}")
            exit(1)
    else:
        logging.info("✅ Model already exists, skipping download.")

# Download the model
download_model()

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"🚫 Failed to load model: {str(e)}")
    exit(1)

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
    filename = secure_filename(file.filename)  # Secure filename
    image_path = os.path.join("temp_" + filename)
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

# Route for testing if the server is running
@app.route("/")
def home():
    return "Hello, Railway!"

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Get PORT from Railway environment variable
    app.run(host='0.0.0.0', port=port, debug=False)  # ✅ Listen on all interfaces

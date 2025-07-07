from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# Model paths
SPIRAL_MODEL_PATH = 'spiral_model.h5'
WAVE_MODEL_PATH = 'wave_model.h5'
ENSEMBLE_MODEL_PATH = 'ensemble_model.pkl'

# Image sizes
SPIRAL_SIZE = (256, 256)
WAVE_SIZE = (512, 512)
CLASSES = ['healthy', 'parkinson']

app = Flask(__name__)

# Load models at startup
spiral_model = tf.keras.models.load_model(SPIRAL_MODEL_PATH)
wave_model = tf.keras.models.load_model(WAVE_MODEL_PATH)
ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve static files (if needed)
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

# Helper: preprocess image
def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('L')  # Grayscale
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, H, W, 1)
    return img_array

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'spiral' not in request.files or 'wave' not in request.files:
        return jsonify({'result': 'error', 'message': 'Both images required.'}), 400
    spiral_file = request.files['spiral']
    wave_file = request.files['wave']

    spiral_filename = secure_filename(spiral_file.filename)
    wave_filename = secure_filename(wave_file.filename)
    spiral_path = os.path.join('temp', spiral_filename)
    wave_path = os.path.join('temp', wave_filename)
    os.makedirs('temp', exist_ok=True)
    spiral_file.save(spiral_path)
    wave_file.save(wave_path)

    try:
        # Preprocess images
        spiral_img = preprocess_image(spiral_path, SPIRAL_SIZE)
        wave_img = preprocess_image(wave_path, WAVE_SIZE)

        # Get probabilities from CNNs
        spiral_prob = spiral_model.predict(spiral_img)[0]  # shape (2,)
        wave_prob = wave_model.predict(wave_img)[0]        # shape (2,)

        # Concatenate for ensemble
        meta_input = np.concatenate([spiral_prob, wave_prob]).reshape(1, -1)  # shape (1,4)

        # Predict with ensemble
        pred = ensemble_model.predict(meta_input)[0]
        result = CLASSES[pred]

        print("Spiral model output:", spiral_prob)
        print("Wave model output:", wave_prob)
        print("Ensemble input:", meta_input)
        print("Ensemble prediction:", pred)
    except Exception as e:
        print("Prediction error:", e)
        # Clean up temp files
        os.remove(spiral_path)
        os.remove(wave_path)
        return jsonify({'result': 'error', 'message': str(e)}), 500

    # Clean up temp files
    os.remove(spiral_path)
    os.remove(wave_path)

    print("Returning result:", result)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=5050) 
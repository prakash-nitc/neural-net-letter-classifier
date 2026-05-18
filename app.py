"""
Handwritten Digit Classifier — Web Application
================================================
Flask web app that lets users upload an image or draw on a canvas,
then classifies the digit (0-9) using a 784-128-10 neural network
trained from scratch on MNIST entirely in NumPy.

Run:
    python app.py
    Open http://127.0.0.1:5000
"""

import os
import io
import base64

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

from deep_neural_network import DeepNeuralNetwork

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

MODEL_PATH  = 'model/mnist_dnn_model.npz'
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

if os.path.exists(MODEL_PATH):
    nn = DeepNeuralNetwork.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    nn = DeepNeuralNetwork([784, 128, 10], activation='relu', loss='cross_entropy')
    print(f"WARNING: model not found at {MODEL_PATH}")
    print("Run  python train_mnist.py  first to train and save the model.")


def preprocess_image(image_data):
    """
    Preprocess an uploaded/drawn image for the MNIST network.

    MNIST convention: dark background (≈-1), bright digit (≈+1).
    Canvas convention: white background, black strokes — so we invert.

    Pipeline:
        Open → grayscale → binary threshold → invert → crop to bounding box
        → pad 20% → resize to 28×28 (LANCZOS) → normalize to [-1, 1] → flatten

    Returns:
        flat_input  : np.ndarray shape (1, 784)
        img_display : PIL Image (28×28), values scaled to [0, 255]
    """
    img = Image.open(io.BytesIO(image_data))

    # Composite onto white background to flatten any transparency
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert('RGBA'))
    img = img.convert('L')

    pixels = np.array(img, dtype=np.float64)

    # Hard threshold → binary {0, 255}
    binary = np.where(pixels < 128, 0.0, 255.0)

    # Invert so that drawn strokes become bright (matching MNIST convention)
    inverted = 255.0 - binary

    # Find bounding box of bright (stroke) pixels
    bright_rows = np.any(inverted > 127, axis=1)
    bright_cols = np.any(inverted > 127, axis=0)

    if not np.any(bright_rows):
        # Nothing drawn — return blank (all background)
        flat = np.full((1, 784), -1.0)
        display = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
        return flat, display

    r_min, r_max = np.where(bright_rows)[0][[0, -1]]
    c_min, c_max = np.where(bright_cols)[0][[0, -1]]
    cropped = inverted[r_min:r_max + 1, c_min:c_max + 1]

    # Add padding (~20% on each side) so the digit doesn't touch edges
    h, w = cropped.shape
    pad_h = max(4, int(h * 0.20))
    pad_w = max(4, int(w * 0.20))
    padded = np.pad(cropped, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=0.0)

    # Resize to 28×28 with anti-aliasing (matches MNIST scale)
    pil_crop = Image.fromarray(padded.astype(np.uint8), mode='L')
    resized  = pil_crop.resize((28, 28), Image.LANCZOS)
    arr28    = np.array(resized, dtype=np.float64)

    # Normalize to [-1, 1]  (same as training)
    normalized = arr28 / 127.5 - 1.0

    # Build display image (rescale back to [0, 255])
    display_pixels = np.clip((normalized + 1.0) * 127.5, 0, 255).astype(np.uint8)
    display = Image.fromarray(display_pixels, mode='L')

    flat = normalized.flatten().reshape(1, -1)
    return flat, display


def _build_response(X, img_display):
    predicted_class, probs = nn.predict(X)
    predicted_digit = CLASS_NAMES[predicted_class[0]]

    confidence = {CLASS_NAMES[i]: float(probs[0][i]) for i in range(10)}

    # Scale up for display
    img_big = img_display.resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    img_big.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        'prediction':         predicted_digit,
        'confidence':         confidence,
        'preprocessed_image': f'data:image/png;base64,{img_b64}',
    })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        image_data = request.files['file'].read()
        X, img_display = preprocess_image(image_data)
        return _build_response(X, img_display)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    try:
        data     = request.get_json()
        img_b64  = data.get('image', '')
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        image_data = base64.b64decode(img_b64)
        X, img_display = preprocess_image(image_data)
        return _build_response(X, img_display)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Handwritten Digit Classifier - Web App")
    print("  Model: 784-128-10  trained on MNIST (NumPy)")
    print("=" * 50)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)

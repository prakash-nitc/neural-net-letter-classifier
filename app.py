"""
Web Application for Letter Classification — Part (d)
=====================================================
Flask web app that allows users to upload an image of B, 0, or E,
preprocesses it to 8×8 pixels, and uses the trained 64-16-3 neural
network to predict the letter with confidence scores.
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

from neural_network import NeuralNetwork

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load the trained model
MODEL_PATH = 'model/model_64_16_3.npz'
CLASS_NAMES = ['B', '0', 'E']

nn = NeuralNetwork(input_size=64, hidden_size=16, output_size=3)

# Try to load the model if it exists
if os.path.exists(MODEL_PATH):
    nn.load(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print("Please run train_and_visualize.py first to train the model.")


def preprocess_image(image_data):
    """
    Preprocess an uploaded image for the neural network.
    
    The model was trained on 8x8 binary images where:
      - Black (letter stroke) = -1.0
      - White (background)    = +1.0
    
    The training templates place the letter in a 7x7 active area (rows 0-6,
    cols 0-6), with row 7 and the rightmost column(s) as white padding.
    
    To match this layout, we:
    1. Convert to grayscale and threshold to binary
    2. Crop to the bounding box of the letter (remove excess whitespace)
    3. Resize the cropped letter to fit a 7x7 area
    4. Place it in an 8x8 grid with white padding on row 7 and col 7
    5. Flatten to a 64-dimensional vector
    
    Args:
        image_data: Image file data (bytes).
    
    Returns:
        tuple: (flat_input, img_display)
            flat_input: np.ndarray of shape (1, 64)
            img_display: PIL Image (8x8) for display
    """
    # Open image and handle transparency
    img = Image.open(io.BytesIO(image_data))
    
    # Critical: PNG images often have transparent backgrounds.
    # PIL's convert('L') treats transparent pixels as BLACK,
    # which destroys the preprocessing (entire image becomes black).
    # Fix: composite onto white background before grayscale conversion.
    if img.mode == 'RGBA' or img.mode == 'LA':
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img.convert('RGBA'))
    img = img.convert('L')
    
    # Step 1: Threshold to binary at full resolution
    pixels_full = np.array(img, dtype=np.float64)
    threshold = 128
    pixels_binary = np.where(pixels_full < threshold, 0, 255).astype(np.uint8)
    
    # Step 2: Find bounding box of the letter (black pixels)
    # Identify rows and columns that contain any black pixels
    black_rows = np.any(pixels_binary == 0, axis=1)
    black_cols = np.any(pixels_binary == 0, axis=0)
    
    if not np.any(black_rows):
        # No black pixels found — return all-white grid
        normalized = np.ones((8, 8))
        flat = normalized.flatten().reshape(1, -1)
        display_pixels = np.full((8, 8), 255, dtype=np.uint8)
        img_display = Image.fromarray(display_pixels, mode='L')
        return flat, img_display
    
    # Get bounding box coordinates
    row_min, row_max = np.where(black_rows)[0][[0, -1]]
    col_min, col_max = np.where(black_cols)[0][[0, -1]]
    
    # Crop to bounding box
    cropped = pixels_binary[row_min:row_max+1, col_min:col_max+1]
    
    # Step 3: Resize the cropped letter to fit a 7x7 area
    # This maps the letter to the same spatial region as the training templates
    crop_img = Image.fromarray(cropped, mode='L')
    crop_resized = crop_img.resize((7, 7), Image.BOX)
    
    # Threshold after resize to get clean binary values
    crop_pixels = np.array(crop_resized, dtype=np.float64)
    crop_binary = np.where(crop_pixels < threshold, -1.0, 1.0)
    
    # Step 4: Place in 8x8 grid with white padding
    # Row 7 = white, Col 7 = white (matching training template layout)
    normalized = np.ones((8, 8))  # All white (+1.0)
    normalized[0:7, 0:7] = crop_binary  # Place letter in top-left 7x7
    
    # Create display version
    display_pixels = np.where(normalized == -1.0, 0, 255).astype(np.uint8)
    img_display = Image.fromarray(display_pixels, mode='L')
    
    # Flatten to 64-dimensional vector
    flat = normalized.flatten().reshape(1, -1)
    
    return flat, img_display


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for image classification.
    
    Accepts an image file upload, preprocesses it, runs the model,
    and returns the prediction with confidence scores.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess the image
        image_data = file.read()
        X, img_resized = preprocess_image(image_data)
        
        # Get prediction
        predicted_class, probabilities = nn.predict(X)
        
        # Build response
        predicted_letter = CLASS_NAMES[predicted_class[0]]
        confidence_scores = {
            CLASS_NAMES[i]: float(probabilities[0][i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Scale up the preprocessed 8x8 display image for visibility
        img_pil = img_resized.resize((128, 128), Image.NEAREST)
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        preprocessed_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'prediction': predicted_letter,
            'confidence': confidence_scores,
            'preprocessed_image': f'data:image/png;base64,{preprocessed_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """
    API endpoint for canvas drawing classification.
    
    Accepts a base64-encoded canvas image, preprocesses it, and
    returns the prediction with confidence scores.
    """
    try:
        data = request.get_json()
        image_b64 = data.get('image', '')
        
        # Remove the data URL prefix
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        # Decode base64 to bytes
        image_data = base64.b64decode(image_b64)
        
        # Preprocess
        X, img_resized = preprocess_image(image_data)
        
        # Predict
        predicted_class, probabilities = nn.predict(X)
        
        predicted_letter = CLASS_NAMES[predicted_class[0]]
        confidence_scores = {
            CLASS_NAMES[i]: float(probabilities[0][i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Scale up the preprocessed 8x8 display image for visibility
        img_pil = img_resized.resize((128, 128), Image.NEAREST)
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        preprocessed_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'prediction': predicted_letter,
            'confidence': confidence_scores,
            'preprocessed_image': f'data:image/png;base64,{preprocessed_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("TFML Assignment 1 — Letter Classifier Web App")
    print("=" * 50)
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)

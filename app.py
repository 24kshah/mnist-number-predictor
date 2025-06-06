from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import base64
from PIL import Image, ImageOps
import io

from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("mnist_cnn_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']

    # Extract base64 string from data URL
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    img_bytes = base64.b64decode(img_str)

    # Open image and convert to grayscale
    img = Image.open(io.BytesIO(img_bytes)).convert('L')

    # Invert image (MNIST digits are white on black)
    img = ImageOps.invert(img)

    # Convert to numpy array
    img_array = np.array(img)

    # Threshold image to binary (black and white)
    threshold = 50
    img_array = (img_array > threshold) * 255

    # Find bounding box of the digit
    coords = np.column_stack(np.where(img_array > 0))
    if coords.size == 0:
        # No drawing detected
        return jsonify({'prediction': None, 'error': 'No digit found. Please draw a digit.'})

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop to bounding box
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]

    # Resize while maintaining aspect ratio and pad to 28x28
    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    cropped_img = Image.fromarray(cropped.astype(np.uint8))
    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create 28x28 black canvas
    final_img = Image.new('L', (28, 28), color=0)
    upper_left_x = (28 - new_w) // 2
    upper_left_y = (28 - new_h) // 2
    final_img.paste(resized_img, (upper_left_x, upper_left_y))

    # Normalize pixels
    final_array = np.array(final_img) / 255.0

    # Reshape for model input
    final_array = final_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(final_array)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")
model = load_model('mnist_cnn_model.h5')

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('image')
        if file:
            image = Image.open(file.stream)
            img = preprocess_image(image)
            prediction = model.predict(img)
            digit = np.argmax(prediction)
            return render_template('index.html', digit=digit)
    return render_template('index.html')

# Vercel requires this to export the Flask app as 'app'
app = app

import os
# Disable GPU to avoid CUDA errors on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Model download + load
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "InceptionV3_model_Adam_Tea.h5")

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1G2QICI8PudXjtsRNIRM2mTYhoapy9S9H"
    gdown.download(url, MODEL_PATH, quiet=False)

# model = tf.keras.models.load_model(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# Class labels
# -----------------------------
class_names = [
    'algal_spot',
    'brown_blight',
    'gray_blight',
    'healthy',
    'helopeltis',
    'red_spot'
]

# -----------------------------
# Recommendations for each disease
# -----------------------------
recommendations = {
    'algal_spot': 'Apply copper-based fungicides (e.g., copper oxychloride). Avoid excessive leaf wetness and improve air circulation. Prune infected areas to limit spread.',
    'brown_blight': 'Remove and destroy affected leaves. Apply protective fungicides like mancozeb or copper oxychloride. Ensure proper drainage and avoid overcrowding of plants.',
    'gray_blight': 'Use systemic fungicides such as thiophanate-methyl or carbendazim. Prune and destroy infected leaves. Maintain field sanitation and avoid excess nitrogen fertilization.',
    'healthy': 'No action needed. Continue regular monitoring and maintain good agronomic practices, including balanced fertilization and pest control.',
    'helopeltis': 'Spray insecticides such as imidacloprid or neem-based formulations. Monitor regularly for nymphs and adults. Use yellow sticky traps and encourage natural predators.',
    'red_spot': 'Apply recommended fungicides like chlorothalonil or mancozeb. Improve drainage and avoid overhead irrigation to reduce leaf wetness. Remove affected leaves.'
}

# -----------------------------
# Preprocess uploaded image
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_filename = None
    recommendation = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_filename = filename

            # Predict
            img = Image.open(filepath)
            processed = preprocess_image(img)
            preds = model.predict(processed)
            pred_index = np.argmax(preds)
            prediction = class_names[pred_index]
            confidence = round(float(np.max(preds)) * 100, 2)
            recommendation = recommendations[prediction]

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_filename=image_filename,
                           recommendation=recommendation)

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

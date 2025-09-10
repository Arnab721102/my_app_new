from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
# model = tf.keras.models.load_model('F://TeaLeaf//InceptionV3_model_Adam_Tea.h5')
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "InceptionV3_model_Adam_Tea.h5"))
print("Model loaded successfully.")
# Define class labels
class_names = [
    'algal_spot',
    'brown_blight', 
    'gray_blight',
    'healthy',
    'helopeltis',
    'red_spot'
]

# Recommendations
recommendations = {
    'algal_spot': 'Apply copper-based fungicides (e.g., copper oxychloride). Avoid excessive leaf wetness and improve air circulation. Prune infected areas to limit spread.',
    'brown_blight': 'Remove and destroy affected leaves. Apply protective fungicides like mancozeb or copper oxychloride. Ensure proper drainage and avoid overcrowding of plants.',
    'gray_blight': 'Use systemic fungicides such as thiophanate-methyl or carbendazim. Prune and destroy infected leaves. Maintain field sanitation and avoid excess nitrogen fertilization.',
    'healthy': 'No action needed. Continue regular monitoring and maintain good agronomic practices, including balanced fertilization and pest control.',
    'helopeltis': 'Spray insecticides such as imidacloprid or neem-based formulations. Monitor regularly for nymphs and adults. Use yellow sticky traps and encourage natural predators.',
    'red_spot': 'Apply recommended fungicides like chlorothalonil or mancozeb. Improve drainage and avoid overhead irrigation to reduce leaf wetness. Remove affected leaves.'
}

# Preprocess input image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

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

if __name__ == '__main__':
    app.run(debug=True)

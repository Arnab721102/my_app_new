import os
import requests
from tensorflow.keras.models import load_model

# Local model filename
MODEL_PATH = "InceptionV3_model_Adam_Tea.h5"

# Google Drive direct download link
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1G2QICI8PudXjtsRNIRM2mTYhoapy9S9H"

def download_model():
    """Download model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        response = requests.get(DRIVE_URL, stream=True)
        response.raise_for_status()  # ensure no silent failures
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")

def get_model():
    """Load and return the trained model."""
    download_model()
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model

# flask_api.py
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "optimized_model.keras"
CLASS_NAMES = [
    'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 
    'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans',
    'VI-chickenpox', 'VI-shingles'
]

# Load model once
model = load_model(MODEL_PATH)

# -----------------------
# Helper
# -----------------------
def preprocess_image(file):
    img = Image.open(file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    html = """
    <h1>🩺 Skin Disease Classifier API</h1>
    <p>Welcome! Your API is running.</p>
    <ul>
        <li><a href="/api/health">/api/health</a> - Check API status</li>
        <li>/api/predict - Use POST with form-data key 'file' to send an image</li>
    </ul>
    """
    return render_template_string(html)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"message": "API is running", "status": "ok"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({
            "error": "No file part",
            "message": "Send an image file with form-data key 'file'"
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        img_array = preprocess_image(file)
        preds = model.predict(img_array, verbose=0)[0]
        top_idx = np.argmax(preds)
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(preds[top_idx])

        return jsonify({
            "top_prediction": top_class,
            "confidence": top_confidence,
            "all_predictions": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import io

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "oral_disease_model.onnx"
CLASS_NAMES = [
    'calculus',
    'caries',
    'gingivitis',
    'mouth_ulcer',
    'tooth_discoloration',
    'hypodontia'
]

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# -----------------------------
# Load ONNX Model (once at startup)
# -----------------------------
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("✅ ONNX model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise SystemExit("Model loading failed. Exiting...")

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: Image.Image):
    """Same preprocessing as Streamlit app (ensures consistency)"""
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image: Image.Image):
    """Run inference and return results"""
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "all_probabilities": {
            name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)
        }
    }

# -----------------------------
# API Endpoints
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🦷 Oral Disease Classification API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Upload an image for diagnosis"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file is in request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Check filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, JPEG, or PNG"}), 400
    
    try:
        # Open and validate image
        image = Image.open(file.stream).convert("RGB")
        
        # Run prediction
        result = predict_image(image)
        
        return jsonify({
            "success": True,
            "prediction": result["predicted_class"].replace('_', ' ').title(),
            "confidence": result["confidence"],
            "all_classes": result["all_probabilities"]
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print(f"🚀 Starting Oral Disease API...")
    print(f"📘 Swagger-like docs: visit http://localhost:5000")
    print(f"🧪 Test with: curl -F 'image=@your_image.jpg' http://localhost:5000/predict")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
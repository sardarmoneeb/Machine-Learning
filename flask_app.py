from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Class information
CLASS_NAMES = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
CLASS_DESCRIPTIONS = {
    'colon_aca': 'Colon Adenocarcinoma (Cancer)',
    'colon_n': 'Colon Normal (Healthy)',
    'lung_aca': 'Lung Adenocarcinoma (Cancer)',
    'lung_n': 'Lung Normal (Healthy)',
    'lung_scc': 'Lung Squamous Cell Carcinoma (Cancer)'
}

# Load model
def load_cancer_model():
    try:
        model = load_model('final_cancer_classifier.h5')
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_cancer_model()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 224x224 (model input size)
    image = image.resize((224, 224))
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_cancer(image):
    """Make prediction on image"""
    try:
        # Preprocess
        processed_img = preprocess_image(image)
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        all_predictions = {}
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions[class_name] = float(predictions[0][i])
        
        return predicted_class, confidence, all_predictions
    except Exception as e:
        raise Exception(f"Prediction error: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "🩺 Lung & Colon Cancer Classification API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Upload image for cancer classification",
            "/health": "GET - API health check"
        },
        "supported_classes": CLASS_DESCRIPTIONS
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "API is running successfully"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file type. Only JPG, JPEG, PNG allowed"}), 400
        
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        predicted_class, confidence, all_predictions = predict_cancer(image)
        
        # Prepare response
        result = {
            "prediction": predicted_class,
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "confidence": confidence,
            "is_cancer": '_n' not in predicted_class,  # True if cancer, False if normal
            "all_predictions": all_predictions,
            "status": "success"
        }
        
        # Add medical advice based on result
        if '_n' in predicted_class:
            result["medical_advice"] = "Normal tissue detected. Regular checkups recommended."
        else:
            result["medical_advice"] = "Abnormal tissue detected. Please consult with a medical professional for proper diagnosis and treatment."
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/predict_url', methods=['POST'])
def predict_from_url():
    """Alternative endpoint for URL-based image upload"""
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({"error": "No image_url provided"}), 400
        
        # This would require additional libraries like requests
        # and handling of URL image downloading
        return jsonify({
            "error": "URL prediction not implemented yet",
            "message": "Please use the /predict endpoint with file upload"
        }), 501
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check if model is loaded
    if model is None:
        print("❌ Cannot start server: Model not loaded")
    else:
        print("🚀 Starting Flask API Server...")
        print("📡 API Endpoints:")
        print("   GET  /          - API information")
        print("   GET  /health    - Health check")
        print("   POST /predict   - Image classification")
        print("\n🎯 Make sure your model file 'final_cancer_classifier.h5' is in the same directory")
        
        # Run the app
        app.run(host='0.0.0.0', port=5000, debug=True)
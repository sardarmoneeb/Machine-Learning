import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷", layout="wide")

# ✅ EXACT 6 classes in the correct order (must match training)
CLASS_NAMES = [
    'calculus',
    'caries',
    'gingivitis',
    'mouth_ulcer',
    'tooth_discoloration',
    'hypodontia'
]

MODEL_PATH = "oral_disease_model.onnx"  # ← Your exported ONNX file

# -----------------------------
# Load ONNX Model
# -----------------------------
@st.cache_resource
def load_onnx_model():
    """Load ONNX model with CPU provider (no GPU needed)"""
    return ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

try:
    session = load_onnx_model()
    st.success("✅ ONNX model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

# -----------------------------
# Image Preprocessing (FIXED: Ensures float32 dtype)
# -----------------------------
def preprocess_image(image: Image.Image):
    """
    Preprocess image to match training pipeline:
    - Resize to 224x224
    - Normalize with ImageNet stats
    - Output dtype: np.float32 (required by ONNX)
    """
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to numpy array and ensure float32
    image = np.array(image).astype(np.float32)
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Apply ImageNet normalization (with explicit float32 dtype)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image: Image.Image):
    """Run inference and return prediction, confidence, and all probabilities"""
    input_tensor = preprocess_image(image)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    logits = outputs[0][0]  # Remove batch dimension
    
    # Apply softmax to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    
    return pred_idx, confidence, probs

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🦷 Oral Disease Classification")
st.write("Upload a dental image to detect one of 6 oral diseases.")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                pred_idx, conf, all_probs = predict(image)
                
                # Display main prediction
                disease_name = CLASS_NAMES[pred_idx].replace('_', ' ').title()
                st.success(f"**Diagnosis:** {disease_name}")
                st.info(f"**Confidence:** {conf:.1%}")
                
                # Display all class probabilities
                st.subheader("📊 Full Diagnosis Report")
                for name, prob in zip(CLASS_NAMES, all_probs):
                    display_name = name.replace('_', ' ').title()
                    st.write(f"**{display_name}**: {prob:.1%}")
                    st.progress(float(prob))
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
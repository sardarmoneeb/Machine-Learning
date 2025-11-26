# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
from PIL import Image
import os
import io

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .disease-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🩺 Skin Disease Classifier</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913502.png", width=100)
    st.title("About")
    st.info("""
    This AI model classifies 8 types of skin diseases with **96.59% accuracy**:
    
    - Bacterial Infections
    - Fungal Infections  
    - Viral Infections
    - Parasitic Infections
    """)
    
    st.markdown("---")
    st.subheader("Model Information")
    st.write("**Framework:** TensorFlow/Keras")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Accuracy:** 96.59%")
    st.write("**Classes:** 8 skin diseases")

# Disease information dictionary
DISEASE_INFO = {
    'BA- cellulitis': {
        'type': 'Bacterial Infection',
        'description': 'A common bacterial skin infection that causes redness, swelling, and pain.',
        'symptoms': 'Red area, swelling, pain, warmth, fever',
        'treatment': 'Antibiotics, wound care'
    },
    'BA-impetigo': {
        'type': 'Bacterial Infection', 
        'description': 'A highly contagious skin infection that mainly affects infants and children.',
        'symptoms': 'Red sores, blisters, itching, honey-colored crusts',
        'treatment': 'Topical antibiotics, oral antibiotics'
    },
    'FU-athlete-foot': {
        'type': 'Fungal Infection',
        'description': 'A fungal infection that usually begins between the toes.',
        'symptoms': 'Itching, scaling, cracking, blisters',
        'treatment': 'Antifungal creams, powders'
    },
    'FU-nail-fungus': {
        'type': 'Fungal Infection',
        'description': 'A fungal infection in the fingernails or toenails.',
        'symptoms': 'Discolored nails, thickening, crumbling edges',
        'treatment': 'Antifungal medications, laser treatment'
    },
    'FU-ringworm': {
        'type': 'Fungal Infection',
        'description': 'A contagious fungal infection of the skin or scalp.',
        'symptoms': 'Ring-shaped rash, itching, red scales',
        'treatment': 'Antifungal creams, oral medications'
    },
    'PA-cutaneous-larva-migrans': {
        'type': 'Parasitic Infection',
        'description': 'A skin infection caused by hookworm larvae.',
        'symptoms': 'Itchy red lines, blisters, creeping eruption',
        'treatment': 'Antiparasitic medications'
    },
    'VI-chickenpox': {
        'type': 'Viral Infection',
        'description': 'A highly contagious viral infection causing an itchy rash.',
        'symptoms': 'Fever, itchy rash, blisters, fatigue',
        'treatment': 'Symptom relief, antiviral medications'
    },
    'VI-shingles': {
        'type': 'Viral Infection',
        'description': 'A viral infection that causes a painful rash.',
        'symptoms': 'Pain, burning, red rash, blisters',
        'treatment': 'Antiviral medications, pain relievers'
    }
}

# Class names (MUST MATCH YOUR TRAINING ORDER)
CLASS_NAMES = [
    'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 
    'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans',
    'VI-chickenpox', 'VI-shingles'
]

@st.cache_resource
def load_skin_model():
    """Load the trained skin disease model"""
    try:
        model = load_model('optimized_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(uploaded_file):
    """Preprocess the uploaded image for prediction"""
    try:
        # Load image
        img = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to match model input
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def predict_disease(model, image_array):
    """Make prediction on the preprocessed image"""
    try:
        predictions = model.predict(image_array, verbose=0)[0]
        return predictions
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    # Load model
    model = load_skin_model()
    
    if model is None:
        st.error("Could not load the model. Please check if 'optimized_model.keras' exists.")
        return

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Skin Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a skin image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the affected skin area"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            with st.spinner("🔄 Processing image..."):
                image_array, processed_img = preprocess_image(uploaded_file)
                
            if image_array is not None:
                # Make prediction
                with st.spinner("🔍 Analyzing skin condition..."):
                    predictions = predict_disease(model, image_array)
                
                if predictions is not None:
                    # Get top prediction
                    top_class_idx = np.argmax(predictions)
                    top_class = CLASS_NAMES[top_class_idx]
                    top_confidence = predictions[top_class_idx]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Diagnosis Results")
                    
                    # Prediction box
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    # Confidence indicator
                    confidence_color = "green" if top_confidence > 0.9 else "orange" if top_confidence > 0.7 else "red"
                    
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Predicted Condition:** {top_class}")
                        st.markdown(f"**Confidence Level:** {top_confidence:.2%}")
                    with col_b:
                        if top_confidence > 0.9:
                            st.success("High Confidence")
                        elif top_confidence > 0.7:
                            st.warning("Medium Confidence")
                        else:
                            st.error("Low Confidence")
                    
                    # Confidence bar
                    st.progress(float(top_confidence))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Disease information
                    if top_class in DISEASE_INFO:
                        info = DISEASE_INFO[top_class]
                        st.markdown('<div class="disease-info">', unsafe_allow_html=True)
                        st.subheader(f"ℹ️ About {top_class}")
                        st.write(f"**Type:** {info['type']}")
                        st.write(f"**Description:** {info['description']}")
                        st.write(f"**Common Symptoms:** {info['symptoms']}")
                        st.write(f"**Typical Treatment:** {info['treatment']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.subheader("📊 All Predictions")
                    top_3_idx = np.argsort(predictions)[-3:][::-1]
                    
                    for i, idx in enumerate(top_3_idx):
                        confidence = predictions[idx]
                        disease_name = CLASS_NAMES[idx]
                        
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(f"{i+1}. {disease_name}")
                        with col2:
                            st.progress(float(confidence))
                        with col3:
                            st.write(f"{confidence:.2%}")
    
    with col2:
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Upload** a clear image of the skin condition
        2. **Wait** for AI analysis (takes 5-10 seconds)
        3. **Review** the diagnosis and confidence level
        4. **Read** about the condition and treatments
        
        **📸 Image Tips:**
        - Use good lighting
        - Focus on affected area
        - Avoid blurry images
        - Include scale reference if possible
        """)
        
        st.markdown("---")
        st.subheader("🏥 Supported Conditions")
        
        # Display all supported diseases
        for i, disease in enumerate(CLASS_NAMES, 1):
            with st.expander(f"{i}. {disease}"):
                if disease in DISEASE_INFO:
                    info = DISEASE_INFO[disease]
                    st.write(f"**Type:** {info['type']}")
                    st.write(f"**Description:** {info['description']}")
        
        st.markdown("---")
        st.subheader("⚠️ Important Disclaimer")
        st.warning("""
        This tool is for educational and informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        
        - Always consult healthcare professionals for medical concerns
        - Do not self-diagnose or self-treat based on AI predictions
        - In case of emergency, seek immediate medical attention
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Skin Disease Classifier | AI-Powered Medical Tool | Accuracy: 96.59%"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
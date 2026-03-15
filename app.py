import streamlit as st
import os
from model.model_loader import load_model
from utils.image_processing import process_image

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Config
st.set_page_config(page_title="OceanGuard AI - Detection Engine", page_icon="🌊")

# Custom Styling
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #0077be; color: white; font-weight: bold; height: 3em;}
    h1 { color: #0077be; text-align: center; }
    .result-card { padding: 20px; border-radius: 15px; border: 2px solid #0077be; background-color: #f0f8ff; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 OceanGuard AI")
st.markdown("### Core Environmental Detection Engine")
st.write("Specialized Vision Transformer (ViT) for SmathHacks Hackathon")

# Load AI Model
@st.cache_resource
def get_ai_model():
    return load_model()

model = get_ai_model()

# Sidebar
with st.sidebar:
    st.header("SmathHacks Submission")
    st.info("Goal: Real-time identification of marine environmental threats.")
    st.write("Leveraging OpenAI CLIP for zero-shot semantic analysis of ocean imagery.")

# Main Detection Logic
uploaded_file = st.file_uploader("Upload Ocean/Coral Imagery", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = process_image(uploaded_file)
    st.image(image, caption="Target Imagery", use_container_width=True)
    
    if st.button("🔍 RUN DETECTION ENGINE"):
        with st.spinner("Neural Network analyzing semantic features..."):
            label, confidence = model.predict(image)
            
            # Clean up label for display
            display_label = label.replace('_', ' ').title()
            
            st.markdown(f"""
            <div class="result-card">
                <h2>Analysis Result: {display_label}</h2>
                <p>Detection Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            if "healthy" in label:
                st.balloons()
                st.success("Environmental Check: Healthy ecosystem detected.")
            else:
                st.error("Environmental Alert: Anomaly/Pollution detected.")
else:
    st.info("Awaiting input imagery for analysis.")

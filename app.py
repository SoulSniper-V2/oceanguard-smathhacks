import streamlit as st
import os
import time
from model.model_loader import load_model
from utils.image_processing import process_image

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Config
st.set_page_config(page_title="OceanGuard AI - Diagnostic Engine", page_icon="🌊")

# Custom Styling
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #0077be; color: white; font-weight: bold; height: 3.5em;}
    h1 { color: #0077be; text-align: center; }
    .diag-box { 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #ddd; 
        background-color: #f8f9fa;
        margin-top: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    .metric { font-size: 1.2em; font-weight: bold; color: #333; }
    .severity-pill { 
        padding: 5px 15px; 
        border-radius: 20px; 
        font-size: 0.9em; 
        background-color: #0077be; 
        color: white; 
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 OceanGuard AI")
st.markdown("### Advanced Environmental Diagnostic Engine")
st.write("Multi-stage Vision AI (BLIP + CLIP) for Deep Classification")

# Load AI Model
@st.cache_resource
def get_ai_model():
    return load_model()

with st.spinner("🧠 Initializing Deep Neural Diagnostic Layers..."):
    model = get_ai_model()

# Sidebar
with st.sidebar:
    st.header("SmathHacks Submission")
    st.info("Goal: Multi-Stage Environmental Intelligence.")
    st.write("Current State: Multi-modal Vision Transformer analyzing semantic depth, severity metrics, and diagnostic narratives.")

# Main Detection Logic
uploaded_file = st.file_uploader("Upload Ocean Imagery", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = process_image(uploaded_file)
    st.image(image, caption="Diagnostic Analysis Target", use_container_width=True)
    
    if st.button("🔎 RUN DEEP DIAGNOSTIC ENGINE"):
        with st.spinner("Neural Network calculating semantic similarity scores..."):
            # Multi-Stage Inference
            report = model.predict(image)
            
            if report:
                # 1. Show Primary Narrative
                st.markdown(f"**AI Narrative Insight:** *\"{report['narrative'].capitalize()}\"*")
                
                # 2. Show Deep Diagnostic Report
                st.markdown("---")
                st.subheader("📋 Deep Diagnostic Report")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Specific Threat Profile:**")
                    st.info(f"🧬 {report['type']}")
                
                with col2:
                    st.write("**Environmental Severity:**")
                    # Visual styling for severity
                    sev_color = "red" if "Severe" in report['severity'] or "Critical" in report['severity'] else "orange"
                    st.markdown(f"<span class='severity-pill' style='background-color: {sev_color}'>{report['severity']}</span>", unsafe_allow_html=True)

                st.progress(report['confidence'])
                st.caption(f"Semantic Alignment Confidence: {report['confidence']*100:.2f}%")

                # Actionable Recommendation
                st.divider()
                if "Healthy" in report['type']:
                    st.success("✅ **Recommendation:** Continue passive monitoring. Ecosystem appears intact.")
                else:
                    st.error("🚨 **Recommendation:** Immediate environmental response triggered. Deploy specialized mitigation teams.")
            else:
                st.error("Diagnostic engine failure. Image data corrupted or incompatible.")
else:
    st.info("Upload imagery for a deep environmental diagnostic report.")

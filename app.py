import streamlit as st
import os
import time
from model.model_loader import load_model
from utils.image_processing import process_image

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Config
st.set_page_config(page_title="OceanGuard AI - Generative Engine", page_icon="🌊")

# Custom Styling
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #0077be; color: white; font-weight: bold; height: 3em;}
    h1 { color: #0077be; text-align: center; }
    .ai-bubble { 
        padding: 25px; 
        border-radius: 20px; 
        border: 3px solid #0077be; 
        background-color: white; 
        color: #0077be;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 OceanGuard AI")
st.markdown("### Generative Environmental Intelligence")
st.write("Using Salesforce BLIP to generate natural language descriptions of marine imagery.")

# Load AI Model
@st.cache_resource
def get_ai_model():
    return load_model()

with st.spinner("🤖 Neural Engine waking up... (may take a moment on first run)"):
    model = get_ai_model()

# Sidebar
with st.sidebar:
    st.header("SmathHacks Submission")
    st.info("Goal: Universal detection of marine features and threats.")
    st.write("Unlike traditional classifiers, this generative model can 'see' and describe literally anything in the imagery.")

# Main Detection Logic
uploaded_file = st.file_uploader("Upload Ocean Imagery", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = process_image(uploaded_file)
    st.image(image, caption="Analysis Target", use_container_width=True)
    
    if st.button("👁️ ASK THE AI"):
        with st.spinner("Analyzing semantic features..."):
            # Generative AI returns a caption
            description, _ = model.predict(image)
            
            # Show "The AI Says" Bubble
            st.markdown(f"""
            <div class="ai-bubble">
                " {description.capitalize()} "
            </div>
            """, unsafe_allow_html=True)
            
            # Check for threats in the text
            threat_keywords = ['oil', 'plastic', 'trash', 'bleached', 'dead', 'waste', 'pollution']
            if any(word in description.lower() for word in threat_keywords):
                st.error("🚨 Environmental Alert: The AI has identified a potential threat in the description.")
            elif 'coral' in description.lower() or 'fish' in description.lower():
                st.success("✅ Ecosystem Insight: Living biological features detected.")
            else:
                st.info("Analysis Complete.")
else:
    st.info("Upload imagery to get a generative description from the AI.")

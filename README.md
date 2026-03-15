# 🌊 OceanGuard AI - Detection Engine

**OceanGuard AI** is a real-time environmental detection engine built for the **SmathHacks Hackathon**. 

It leverages **Zero-Shot Learning** via OpenAI's CLIP (Vision Transformer) to analyze marine imagery and identify environmental threats with semantic understanding.

## 🚀 Vision
Our goal is to provide a lightweight, high-performance tool that can be deployed on research vessels or satellite stations to automatically flag:
- **Bleached/Dead Coral**
- **Plastic Pollution**
- **Oil Spills**
- **Ecosystem Health (Healthy Coral)**

## 🛠️ Core Technology
- **Model:** `CLIP-ViT-Base-Patch32` (Zero-Shot Learning)
- **Engine:** Python / PyTorch / Transformers
- **UI:** Streamlit (Optimized for SmathHacks Demo)

## 📦 Getting Started

1. **Setup Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Engine:**
   ```bash
   streamlit run app.py
   ```

---
**Hackathon Submission:** SmathHacks  
**Theme:** "Under the Sea" - Marine Environmental Protection

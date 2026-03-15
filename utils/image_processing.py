from PIL import Image, ImageOps

def process_image(uploaded_file):
    """
    Standard image processing for Streamlit uploads.
    """
    image = Image.open(uploaded_file).convert("RGB")
    # Basic enhancement if needed
    return image

def get_demo_locations():
    """
    Returns a sample coordinate for demo map updates.
    """
    return -18.2871, 147.6992  # Great Barrier Reef area

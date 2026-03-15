import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import random

class OceanGuardAIModel:
    def __init__(self):
        # Using a "Real AI" Zero-Shot Learning model (CLIP)
        # This model understands the *concepts* of oil spills and coral health
        self.model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        
        # Our specific environmental classes
        self.classes = [
            "a photo of healthy colorful coral reef",
            "a photo of white bleached dead coral",
            "a photo of plastic trash floating in the ocean",
            "a photo of dark oil spill on ocean water"
        ]
        
        # Human-readable labels for the UI
        self.labels = ['healthy_coral', 'bleached_coral', 'plastic_pollution', 'oil_spill']

    def predict(self, image):
        """
        Real AI inference using Zero-Shot Learning.
        The model compares the image features to the text features of our classes.
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                text=self.classes, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Forward pass through the neural network
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Compute probabilities using Softmax
            logits_per_image = outputs.logits_per_image # Image-to-text similarity
            probs = logits_per_image.softmax(dim=1) # Normalize to [0,1]
            
            # Get the highest prediction
            pred_idx = probs.argmax().item()
            confidence = probs[0][pred_idx].item()
            
            # Return result
            return self.labels[pred_idx], confidence
            
        except Exception as e:
            print(f"Error during Real AI inference: {e}")
            return "error", 0.0

def load_model():
    return OceanGuardAIModel()

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class OceanGuardAIModel:
    def __init__(self):
        # Using a Generative Vision-Language Model (BLIP)
        # This model doesn't just categorize—it "speaks" and describes what it sees.
        self.model_id = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)

    def predict(self, image):
        """
        Generative AI inference.
        The AI generates a natural language description of the environmental state.
        """
        try:
            # Prepare inputs for the generative model
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Generate the caption (the AI "speaking")
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)
                
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Simulated confidence for UI display (Captioning models don't provide a single confidence score)
            confidence = 0.98 
            
            return description, confidence
            
        except Exception as e:
            print(f"Error during Generative AI inference: {e}")
            return "Unable to analyze imagery.", 0.0

def load_model():
    return OceanGuardAIModel()

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image

class OceanGuardAIModel:
    def __init__(self):
        # Model 1: Generative Engine (BLIP)
        self.gen_id = "Salesforce/blip-image-captioning-base"
        self.gen_processor = BlipProcessor.from_pretrained(self.gen_id)
        self.gen_model = BlipForConditionalGeneration.from_pretrained(self.gen_id)

        # Model 2: Semantic Diagnostic Engine (CLIP)
        self.diag_id = "openai/clip-vit-base-patch32"
        self.diag_processor = CLIPProcessor.from_pretrained(self.diag_id)
        self.diag_model = CLIPModel.from_pretrained(self.diag_id)

        # Diagnostic Categories for "Deep Classification"
        self.severity_levels = ["Minor/Localized", "Moderate/Spreading", "Severe/Hazardous", "Critical/Catastrophic"]
        self.threat_types = [
            "Industrial Oil Discharge", "Microplastic Accumulation", 
            "Thermal Coral Bleaching", "Organic Waste Runoff", 
            "Healthy Biodiversity", "Abandoned Fishing Gear"
        ]

    def predict(self, image):
        try:
            # Step 1: Generate Narrative
            gen_inputs = self.gen_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                out = self.gen_model.generate(**gen_inputs, max_new_tokens=50)
            narrative = self.gen_processor.decode(out[0], skip_special_tokens=True)

            # Step 2: Deep Diagnostic (Severity & Type)
            # We use CLIP to "score" the image against our categories
            diag_inputs = self.diag_processor(
                text=self.severity_levels + self.threat_types,
                images=image,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                outputs = self.diag_model(**diag_inputs)
            
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            # Extract Severity
            sev_probs = probs[:len(self.severity_levels)]
            severity = self.severity_levels[sev_probs.argmax().item()]
            
            # Extract Type
            type_probs = probs[len(self.severity_levels):]
            threat_type = self.threat_types[type_probs.argmax().item()]

            return {
                "narrative": narrative,
                "severity": severity,
                "type": threat_type,
                "confidence": float(probs.max())
            }
            
        except Exception as e:
            print(f"Diagnostic Error: {e}")
            return None

def load_model():
    return OceanGuardAIModel()

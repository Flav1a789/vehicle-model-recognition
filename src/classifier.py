from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
import cv2

class VehicleClassifier:
    """
    Classifies vehicle model using Stanford Cars dataset model
    (recognizes 196 vehicle types)

    - classify
    - simplify_name
    """
    
    def __init__(self, model_name="therealcyberlord/stanford-car-vit-patch16"):

        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # bug! check if device and data .to cuda
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  

    def classify(self, frame, bbox, min_size=50):
        """
        Returns:
            Dict with:
                 label
                 confidence for label
        """
        x1, y1, x2, y2 = bbox
        
        # Checking if crop is large enough for classification,problem tp detect it in traffic video
        width = x2 - x1
        height = y2 - y1
        
        if width < min_size or height < min_size:
            return {
                'label': 'Vehicle too small to detect',
                'confidence': 0.0
            }
        
        #bug, sometimes YOLO returns negative or out of frame coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return {
                'label': 'Invalid crop',
                'confidence': 0.0
            }
        
        # Different formats: BGR (OpenCV) to RGB (PIL) format for model
        vehicle_crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(vehicle_crop_rgb)
        
        # Prepare inputs for the model
        inputs = self.extractor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run in inference mode
        with torch.no_grad():
             # outputs = model(pixel_values=inputs['pixel_values'], 
             #     attention_mask=inputs['attention_mask'])
            
            outputs = self.model(**inputs) 

        #tensor of classes
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        probabilities = torch.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class_idx].item()
        
        label = self.model.config.id2label[predicted_class_idx]
        
        return {
            'label': label,
            'confidence': confidence
        }
    






    def simplify_label(self, label):
        """            
        Returns:
            2012 BMW M3 coupe-> BMW M3
        """
        parts = label.split()
        
        if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) == 4:
            parts = parts[1:]
        
        # first 2-3 words 
        if len(parts) >= 3:
            return ' '.join(parts[:3])
        elif len(parts) >= 2:
            # Keep only brand + model
            return ' '.join(parts[:2])
        
        return label if label else "Unknown"
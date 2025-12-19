from ultralytics import YOLO
import cv2

class VehicleDetector:
    """
    Detects vehicles, using YOLOv8 and COCO dataset
    """
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
      
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # COCO dataset class IDs for vehicles 2: car, 5: bus, 7: truck
        self.vehicle_classes = [2,5,7]


    def detect(self, frame):
        """            
        Returns:
            vehicle[dicts]:
                - bbox: (x1, y1, x2, y2) 
                - confidence
                - class_name: vehicle class
        
        results sturcture in documentation: https://docs.ultralytics.com/modes/predict/#working-with-results
        """
        results = self.model(
            frame, 
            classes=self.vehicle_classes,
            verbose=False  
        )
        vehicles = [] 

        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            
            class_id = int(box.cls[0].item())
            class_name = results[0].names[class_id]
            
            if confidence >= self.confidence_threshold:
                vehicles.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_name': class_name
                })
        
        return vehicles
import cv2
from tqdm import tqdm
import os

class VideoProcessor:
    """
    Putting together vehicle detection + model classification
    - draw_vehicle_info
    - process_video
    """
    
    def __init__(self, detector, classifier, show_confidence=True):

        self.detector = detector
        self.classifier = classifier
        self.show_confidence = show_confidence

    #Form open CV tutorials
    def draw_vehicle_info(self, frame, bbox, label, confidence):
        """        
        Args:
            frame: Video frame in format
            bbox: bounding box coordinates (x1, y1, x2, y2)
            label
            confidence
        """
        x1, y1, x2, y2 = bbox
        
        # Text settings (STANDART from tutorial)

        if self.show_confidence:
            display_text = f"{label} ({confidence:.2f})"
        else:
            display_text = label
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            display_text, font, font_scale, thickness
        )
        
        # (makes text readable on any background)
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            (0, 255, 0), 
            -1  
        )
        
        # Bounding box  and label settings 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(
            frame,
            display_text,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            (0, 0, 0), 
            thickness
        )
        
        return frame
    
    def process_video(self, input_path, output_path):

        capture = cv2.VideoCapture(input_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # properties
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0 
        with tqdm(total=total_frames, desc="Progress", unit="frame") as pbar:
            while capture.isOpened():
                ret, frame = capture.read()
                #checkpoint
                if not ret:
                    break 

                # Detection
                vehicles = self.detector.detect(frame)
                
                # Classification
                for vehicle in vehicles:
                    bbox = vehicle['bbox']
                    classification = self.classifier.classify(frame, bbox)
                    
                    label = self.classifier.simplify_label(
                        classification['label']
                    )
                    
                    
                    #frame draw
                    frame = self.draw_vehicle_info(
                        frame, bbox, label, classification['confidence']
                    )

# add simplification
                    
                
                out.write(frame)
                
                frame_count += 1
                processed_count += 1
                pbar.update(1)
        
        capture.release()
        out.release()

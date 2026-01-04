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
        self.tracker = VehicleTracker(confidence_increase_threshold = 1.20)



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

                # Detection with tracking
                vehicles = self.detector.detect(frame)
                
                # Classification
                for vehicle in vehicles:
                    bbox = vehicle['bbox']
                    track_id = vehicle['track_id']

                    classification = self.classifier.classify(frame, bbox)
                    simplified_label = self.classifier.simplify_label(
                            classification['label'])
                            
                    stable_label = self.tracker.update_vehicle(
                        track_id=track_id,
                        new_label= simplified_label,
                        new_confidence=classification['confidence']
                        )
                    frame = self.draw_vehicle_info(
                        frame, bbox, stable_label, classification['confidence']
                    )

                out.write(frame)
                frame_count += 1
                processed_count += 1
                pbar.update(1)

                #Interesting suggestion from Claude for Optimization: Memory Cleanup every 30 frames
                #Do we use it in real applications and what are the effects?
                if frame_count % 30 == 0:  # ← 
                active_ids = {v['track_id'] for v in vehicles if v['track_id'] != -1}
                self.tracker.clear_old_vehicles(active_ids)
        
        capture.release()
        out.release()

class VehicleTracker:
    """Track vehicles"""

    def __init__(self, confidence_increase_threshold= 1.20):
        self.threshold = confidence_increase_threshold
        self.vehicle_history = {}
    
    def update_vehicles(self, track_id, new_label, new_confidence):
        """
        Method updates vehicle data only if new label&confidence pass threshold

        Args: 
            track_id
            new_label
            new_confidence
        Returns:
            dict{'label':str, 'confidence':float}

        """
        #Untracked vehicle
        if track_id == -1:
            return 
            

        #Tracked vehicle
        if track_id not in self.vehicle_history:
        #Seen for the first time
            self.vehicle_history[track_id]= {
                'label':new_label,
                'confidence': new_confidence
            }
            return {
                'label':new_label,
                'confidence': new_confidence
            }
        
        #Seen prevoiusly
        old_info = self.vehicle_history[track_id]
        old_label = old_info['label']
        old_confidence = old_info['confidence']

        #Check if label has changed
        if new_label == old_label:
            #keep max confidence
            updated_confidence = max(old_confidence, new_confidence)
                
            self.vehicle_history[track_id]['confidence']= updated_confidence
            
            return{
                'label':new_label,
                'confidence': new_confidence
            }
        #Checking if new_confidence passes the threshold, then update vehicle
        required_confidence = old_confidence * self.threshold
        if new_confidence>= required_confidence:
            self.vehicle_history[track_id] = {
            'label': new_label,
            'confidence': new_confidence
            }
            return {
                'label': new_label,
                'confidence': new_confidence
            }
        #else: keep old label
        else:
            return {
                'label': old_label,
                'confidence': old_confidence
            }

    def get_vehicle_info(self):
        """
        store vehicle info
        """
        return self.vehicle_history.get(track_id, None)
        
    def clear_old_vehicles(self):
        """for better memory management
        should clarify its use
        Removes vehices if not active in set?"""

        self.vehicle_history= {
            tid:info
            for tid, info in self.vehicle_history.items()
            if tid in active_track_ids
        }
        
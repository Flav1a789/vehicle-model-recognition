
from src.detector import VehicleDetector
from src.classifier import VehicleClassifier
from src.video_prcessor import VideoProcessor

import traceback
import os
import sys

def main():

    INPUT_VIDEO = 'input/traffic.mp4'
    OUTPUT_VIDEO = 'output/traffic_output_video.mp4'# Name changing depending on the name of input video
    
    DETECTION_CONFIDENCE = 0.5  # Minimum confidence accepted vehicle detection

    os.makedirs('output', exist_ok=True)
    

    #
    try:
        detector = VehicleDetector(
            model_name='yolov8n.pt',
            confidence_threshold=DETECTION_CONFIDENCE
        )
        
        classifier = VehicleClassifier(
            model_name="therealcyberlord/stanford-car-vit-patch16"
        )
        
        processor = VideoProcessor(
            detector=detector,
            classifier=classifier,
            show_confidence=True
        )
        
        processor.process_video(
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
        )

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
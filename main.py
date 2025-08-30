# -*- coding: utf-8 -*-
"""
Final Ambulance Detection System
- Uses a custom-trained YOLOv8 model for high-accuracy detection.
"""

import cv2
import os
import time
import threading
from collections import deque
import logging

# Make sure you have installed the necessary packages:
# pip install opencv-python ultralytics

from ultralytics import YOLO

class AmbulanceDetectionSystem:
    def __init__(self, model_path, confidence_threshold=0.7):
        """
        Initializes the system with a custom-trained model.

        Args:
            model_path (str): Path to the custom-trained YOLOv8 model (best.pt).
            confidence_threshold (float): Minimum confidence for a detection to be valid.
        """

        # Load the custom-trained YOLOv8 model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.emergency_active = False
        self.traffic_signal_state = "RED"
        
        # Setup for logging system events
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Dictionary to hold system statistics
        self.stats = {
            'total_detections': 0,
            'signal_changes': 0,
        }
        
    def detect_in_frame(self, frame):
        """
        Detects ambulances in a single video frame.

        Args:
            frame: An image frame from the video stream.

        Returns:
            A list of dictionaries, where each dictionary represents a detected ambulance.
        """
        # Perform inference on the frame
        results = self.model(frame)
        ambulance_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf)
                    # Check if the confidence is above our threshold
                    if confidence > self.confidence_threshold:
                        class_id = int(box.cls)
                        class_name = self.model.names[class_id]
                        
                        # Check if the detected object is an ambulance
                        if class_name.lower() == 'ambulance':
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                            }
                            ambulance_detections.append(detection)
                            
        return ambulance_detections
    
    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels on the frame for visualization.
        """
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw the bounding box rectangle in red
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            # Create the label text
            label = f"AMBULANCE {confidence:.2f}"
            
            # Draw a filled background for the label for better visibility
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                          (bbox[0] + label_size[0], bbox[1] - 5), (0, 0, 255), -1)
            
            # Put the label text on the frame
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame
    
    def trigger_emergency_protocol(self):
        """
        Activates the emergency state: turns the signal GREEN and starts a reset timer.
        """
        if not self.emergency_active:
            self.emergency_active = True
            self.traffic_signal_state = "GREEN"
            self.stats['signal_changes'] += 1
            
            self.logger.info("🚨 EMERGENCY PROTOCOL ACTIVATED - Signal changed to GREEN")
            
            # Use a non-blocking timer to reset the protocol after 30 seconds
            threading.Timer(30.0, self.reset_emergency_protocol).start()
    
    def reset_emergency_protocol(self):
        """
        Resets the emergency state back to normal (RED signal).
        """
        self.emergency_active = False
        self.traffic_signal_state = "RED"
        self.logger.info("✅ Emergency protocol reset - Signal returned to normal")
    
    def process_video_stream(self, source=0):
        """
        Main loop to capture video, process frames, and display the output.
        
        Args:
            source (int or str): Video source. 0 for webcam, or a path to a video file.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return
        
        self.logger.info("Starting video processing with custom model...")
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("End of video stream.")
                    break
                
                start_time = time.time()
                
                detections = self.detect_in_frame(frame)
                
                if detections:
                    self.stats['total_detections'] += len(detections)
                    self.trigger_emergency_protocol()
                
                frame = self.draw_detections(frame, detections)
                self.add_info_overlay(frame)
                
                # Calculate and display FPS
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time
                fps_counter.append(fps)
                avg_fps = sum(fps_counter) / len(fps_counter)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Ambulance Detection System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Video processing stopped.")
    
    def add_info_overlay(self, frame):
        """
        Adds a semi-transparent overlay with system status information.
        """
        height, width, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 300, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        status_color = (0, 255, 0) if self.emergency_active else (255, 255, 255)
        info_text = [
            f"Detections: {self.stats['total_detections']}",
            f"Signal Changes: {self.stats['signal_changes']}",
            f"Emergency: {'ACTIVE' if self.emergency_active else 'NORMAL'}",
            f"Signal: {self.traffic_signal_state}"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 35 + i * 25
            color = status_color if "ACTIVE" in text or "GREEN" in text else (255, 255, 255)
            cv2.putText(frame, text, (width - 290, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    """
    Main function to run the ambulance detection system.
    """
    print("🚀 Initializing Ambulance Detection System...")
    
    # --- IMPORTANT: UPDATE THIS LINE ---
    # This should be the name of your model file if it's in the same folder.
    custom_model_path = "best.pt" 
    
    # Check if the model file exists before starting
    if not os.path.exists(custom_model_path):
        print(f"❌ ERROR: Model file not found at '{custom_model_path}'")
        print("Please make sure your 'best.pt' file is in the same folder as this script.")
        return

    # Create an instance of the detection system
    detector = AmbulanceDetectionSystem(model_path=custom_model_path)
    
    print("✅ System initialized successfully!")
    print("📹 Starting video processing... Press 'q' in the video window to quit.")
    
    # Start processing the video from the default webcam (source=0)
    # You can also use a video file: source="path/to/your/video.mp4"
    detector.process_video_stream("data2.mp4")

    print("👋 System shutdown complete.")

if __name__ == "__main__":
    main()

# video_analyzer.py - Simplified Implementation (No Counting)
import cv2
import torch
import json
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import os
import threading
import time

# Global dictionary to keep track of active analyzers by sid
analyzers_by_sid = {}

class VideoAnalyzer:
    def __init__(self, config, socketio=None, sid=None):
        self.video_path = config['video_path']
        self.socketio = socketio
        self.sid = sid
        self.model_path = config['model_path']
        self.display_frames = config.get('display_frames', True)
        self.results_dir = "results"
        self.show_window = config.get('show_window', True)  # For local display
        os.makedirs(self.results_dir, exist_ok=True)
        
        if sid:
            analyzers_by_sid[sid] = self

        self.conf_threshold = config.get('conf_threshold', 0.2)  # Changed to 0.2
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.resize_width = config.get('resize_width', 1280)
        self.resize_height = config.get('resize_height', 720)

    def set_display_frames(self, value):
        self.display_frames = value

    def process_video(self):
        """
        Process the entire video - Object detection only (no counting)
        """
        if not torch.cuda.is_available():
            print("CUDA not available. Running on CPU.")
            if self.socketio:
                self.socketio.emit('status_update', {'msg': 'Warning: CUDA not found. Processing on CPU.'}, room=self.sid)
        
        # Load YOLO model
        print(f"Loading YOLO model from: {self.model_path}")
        model = YOLO(self.model_path)
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video file: {self.video_path}"
            print(error_msg)
            if self.socketio:
                self.socketio.emit('processing_error', {'msg': error_msg}, room=self.sid)
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use resize dimensions
        frame_width = self.resize_width
        frame_height = self.resize_height
        
        print(f"Original Video: {original_width}x{original_height}")
        print(f"Resized to: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        output_video_name = f"processed_{os.path.splitext(os.path.basename(self.video_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_video_path = os.path.join(self.results_dir, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Initialize tracking variables
        detected_objects = []
        frame_number = 0

        print("Starting video processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            # Process the resized frame
            results = model.track(
                frame, 
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False,
                imgsz=640,
                conf=self.conf_threshold,
                iou=self.iou_threshold
            )

            # Process detections
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Get track IDs if available
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                    # Get class name from model
                    class_name = model.names[class_id]
                    
                    # Get track ID if available
                    track_id = track_ids[i] if track_ids is not None else None
                    
                    # Store detection info
                    detection = {
                        'frame': frame_number,
                        'timestamp_seconds': round(frame_number / fps, 2),
                        'object_class': class_name,
                        'confidence': round(float(conf), 2),
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    }
                    
                    if track_id is not None:
                        detection['track_id'] = int(track_id)
                    
                    detected_objects.append(detection)
                    
                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # Display class name and confidence (and ID if available)
                    if track_id is not None:
                        label = f"{class_name} ID:{track_id} ({conf:.2f})"
                    else:
                        label = f"{class_name} ({conf:.2f})"
                    
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add frame number and progress
            progress_text = f"Frame: {frame_number}/{total_frames}"
            cv2.putText(frame, progress_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            video_writer.write(frame)
            
            # Display frame locally if enabled
            if self.show_window:
                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
            
            # Send frame via SocketIO if available
            if self.display_frames and self.socketio:
                _, buffer = cv2.imencode('.jpg', frame)
                self.socketio.emit('update_frame', {'image': np.asarray(buffer).tobytes()}, room=self.sid)
                self.socketio.sleep(0.01)
            
            frame_number += 1
            
            # Print progress every 100 frames
            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

        # Cleanup
        cap.release()
        video_writer.release()
        if self.show_window:
            cv2.destroyAllWindows()
        
        # Save results to JSON
        json_filename = f"report_{os.path.splitext(os.path.basename(self.video_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = os.path.join(self.results_dir, json_filename)
        
        # Count unique objects by class
        object_summary = {}
        for detection in detected_objects:
            class_name = detection['object_class']
            object_summary[class_name] = object_summary.get(class_name, 0) + 1
        
        results_data = {
            "total_detections": len(detected_objects),
            "object_summary": object_summary,
            "video_info": {
                "path": self.video_path,
                "fps": fps,
                "total_frames": total_frames,
                "original_resolution": f"{original_width}x{original_height}",
                "output_resolution": f"{frame_width}x{frame_height}"
            },
            "processing_config": {
                "model_path": self.model_path,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold
            },
            "detections": detected_objects
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"\nProcessing completed!")
        print(f"Total detections: {len(detected_objects)}")
        print(f"Object summary: {object_summary}")
        print(f"Output video saved: {output_video_path}")
        print(f"Results saved: {json_path}")

        if self.socketio:
            self.socketio.emit('processing_complete', {
                'detections': len(detected_objects),
                'object_summary': object_summary,
                'json_path': json_path,
                'video_path': output_video_path
            }, room=self.sid)

        return {
            'detections': len(detected_objects),
            'object_summary': object_summary,
            'video_path': output_video_path,
            'json_path': json_path
        }


# Simple standalone usage example
def run_video_analysis(video_path, model_path):
    """
    Simple function to run video analysis without SocketIO
    """
    config = {
        'video_path': video_path,
        'model_path': model_path,
        'display_frames': False,  # Not using SocketIO
        'show_window': True,      # Show local window
        'conf_threshold': 0.2,    # Changed to 0.2
        'iou_threshold': 0.45
    }
    
    analyzer = VideoAnalyzer(config)
    results = analyzer.process_video()
    return results


# Example usage and testing
if __name__ == "__main__":
    # Fixed configuration with your specific paths - NO USER INPUT REQUIRED
    model_path = r'C:\Users\admin\Downloads\Gunny_bags_detection_using_yolov11\multi_object_code\video_analyzer\multi_object_detector.pt'
    video_path = r'C:\Users\admin\Downloads\Gunny_bags_detection_using_yolov11\multi_object_code\video_analyzer\gudivada_ch3_20250619152123_20250619172534_clip.mp4'
    
    config = {
        'video_path': video_path,
        'model_path': model_path,
        'display_frames': False,
        'show_window': True,
        'conf_threshold': 0.2,    # Set to 0.2 as requested
        'iou_threshold': 0.45,
        'resize_width': 1280,
        'resize_height': 720
    }
    
    print("="*60)
    print("STARTING VIDEO ANALYSIS")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence Threshold: {config['conf_threshold']}")
    print("Press 'q' in the video window to stop processing")
    print("="*60)
    
    try:
        analyzer = VideoAnalyzer(config)
        results = analyzer.process_video()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total detections: {results['detections']}")
        print(f"Object summary: {results['object_summary']}")
        print(f"Output video: {results['video_path']}")
        print(f"Results JSON: {results['json_path']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
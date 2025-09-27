# video_analyzer.py
import cv2
import torch
import json
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import os

# Global dictionary to keep track of active analyzers by sid
analyzers_by_sid = {}

class VideoAnalyzer:
    def __init__(self, config, socketio, sid):
        self.video_path = config['video_path']
        self.roi = config.get('roi')
        self.line_coords = config['line']
        self.process_full_frame = config['process_full']
        self.socketio = socketio
        self.sid = sid  # Store the session id for targeted emits
        self.model_path = config['model_path']
        self.display_frames = config.get('display_frames', True)
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        analyzers_by_sid[sid] = self
        
        arrow_start = np.array(config['direction_arrow'][0])
        arrow_end = np.array(config['direction_arrow'][1])
        self.direction_vector = arrow_end - arrow_start

        # FIXED: Changed default confidence to a more reasonable value
        self.conf_threshold = 0.2
        self.iou_threshold = 0.45

    def set_display_frames(self, value):
        self.display_frames = value

    def _get_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _check_line_crossing(self, point, prev_point, line_p1, line_p2):
        def get_side(p, line_start, line_end):
            val = (p[0] - line_start[0]) * (line_end[1] - line_start[1]) - \
                  (p[1] - line_start[1]) * (line_end[0] - line_start[0])
            if val == 0: return 0
            return 1 if val > 0 else -1
        
        prev_side = get_side(prev_point, line_p1, line_p2)
        current_side = get_side(point, line_p1, line_p2)

        if prev_side * current_side < 0:
            bag_movement_vector = np.array(point) - np.array(prev_point)
            dot_product = np.dot(bag_movement_vector, self.direction_vector)
            
            if dot_product > 0:
                return True
        
        return False

    def process_video(self):
        """
        UPDATED: This function has been rewritten for more robust detection.
        - Always processes the full frame for better model accuracy.
        - Explicitly sets `imgsz` and a higher `conf` threshold.
        - Filters results by ROI after detection if an ROI is used.
        """
        if not torch.cuda.is_available():
            print("CUDA not available. Running on CPU.")
            self.socketio.emit('status_update', {'msg': 'Warning: CUDA not found. Processing on CPU.'}, room=self.sid)
        
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.socketio.emit('processing_error', {'msg': 'Could not open video file.'}, room=self.sid)
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_video_name = f"processed_{os.path.splitext(os.path.basename(self.video_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_video_path = os.path.join(self.results_dir, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        count = 0
        track_history = {}
        counted_ids = set()
        crossing_events = []
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Always process the full frame for better detection accuracy
            frame_to_process = frame

            # --- START OF MAJOR CHANGES ---
            results = model.track(
                frame_to_process, 
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False,
                imgsz=640,                  # FIXED: Explicitly set the image size for consistency.
                conf=self.conf_threshold,   # FIXED: Using the class's conf_threshold (0.4).
                iou=self.iou_threshold
            )
            # --- END OF MAJOR CHANGES ---

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    box_orig = box
                    center = self._get_center(box_orig)

                    # NEW: If using an ROI, check if the object's center is inside it before proceeding.
                    if not self.process_full_frame and self.roi:
                        rx, ry, rw, rh = self.roi
                        if not (rx < center[0] < rx + rw and ry < center[1] < ry + rh):
                            continue # Skip this object, it's outside the defined ROI

                    prev_point = track_history.get(track_id)
                    
                    if prev_point and track_id not in counted_ids:
                        line_p1 = tuple(self.line_coords[0])
                        line_p2 = tuple(self.line_coords[1])
                        
                        if self._check_line_crossing(center, prev_point, line_p1, line_p2):
                            count += 1
                            counted_ids.add(track_id)
                            timestamp = frame_number / fps
                            event = {'bag_id': int(track_id), 'timestamp_seconds': round(timestamp, 2)}
                            crossing_events.append(event)
                            self.socketio.emit('count_update', {'count': count}, room=self.sid)

                    track_history[track_id] = center
                    
                    cv2.rectangle(frame, (box_orig[0], box_orig[1]), (box_orig[2], box_orig[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (box_orig[0], box_orig[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # --- Drawing and emitting frame ---
            if not self.process_full_frame and self.roi:
                 cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[0]+self.roi[2], self.roi[1]+self.roi[3]), (255, 0, 0), 2) # Draw ROI
            cv2.line(frame, tuple(self.line_coords[0]), tuple(self.line_coords[1]), (0, 0, 255), 3) # Draw Line
            cv2.putText(frame, f"Bags Counted: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
            cv2.putText(frame, f"Bags Counted: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            video_writer.write(frame)
            if self.display_frames:
                _, buffer = cv2.imencode('.jpg', frame)
                self.socketio.emit('update_frame', {'image': np.asarray(buffer).tobytes()}, room=self.sid)
            self.socketio.sleep(0.01)
            frame_number += 1

        cap.release()
        video_writer.release()
        
        json_filename = f"report_{os.path.splitext(os.path.basename(self.video_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = os.path.join(self.results_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump({ "total_count": count, "events": crossing_events }, f, indent=4)

        self.socketio.emit('processing_complete', {
            'count': count, 
            'json_path': json_path,
            'video_path': output_video_path
        }, room=self.sid)
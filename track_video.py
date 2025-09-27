import cv2
from ultralytics import YOLO
import tkinter as tk
from pathlib import Path
import os
import requests
import csv
import logging
import re
import json
import numpy as np
import math
import time # Import time for handling API rate limits


OUTPUT_PATH = r"C:\Users\admin\Downloads\Vehicles_ouput"

MODEL_PATH = r"C:\Users\admin\Downloads\analysis code\bikavolu_ch2_20250514105341_20250514105612.pt"
INPUT_VIDEO_PATH = r"C:\Users\admin\Downloads\face_detection\videos\bikavolu_ch2_20250514105341_20250514105612.mp4"
PLATE_RECOGNIZER_API_KEY = "dd80f8560d2878c67da953cd8fd0ab84dc64e040"
CONFIDENCE_THRESHOLD = 0.5
RESIZE_PREVIEW_WINDOW = True
PREVIEW_WINDOW_SCALE = 0.95

# --- QUALITY SCORING CONFIGURATION ---
IDEAL_BRIGHTNESS = 127.0
BRIGHTNESS_WEIGHT = 0.5
SHARPNESS_WEIGHT = 0.2

# --- MOTION DETECTION CONFIGURATION ---
STATIONARY_THRESHOLD_PIXELS = 5
INACTIVITY_SECONDS = 1000

# <<< NEW >>>: Configuration for giving preference to certain state codes
# --- PLATE RECOGNITION ENHANCEMENT ---

PREFERRED_STATE_CODES = ["AP", "TS"] 

PREFERENCE_BONUS = 0.10


os.makedirs(OUTPUT_PATH, exist_ok=True)

log_file_path = Path(OUTPUT_PATH) / 'vehicle_tracker.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path, mode='a'), logging.StreamHandler()])


def get_display_resolution():
    root = tk.Tk(); root.withdraw(); return root.winfo_screenwidth(), root.winfo_screenheight()

def do_boxes_intersect(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA; bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def get_box_area(box):
    x1, y1, x2, y2 = box
    return abs(x2 - x1) * abs(y2 - y1)

def calculate_plate_quality(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return 0, 0, 0
    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_plate)
    brightness_score = 1.0 - (abs(average_brightness - IDEAL_BRIGHTNESS) / IDEAL_BRIGHTNESS)
    sharpness_score = cv2.Laplacian(gray_plate, cv2.CV_64F).var()
    if sharpness_score < 0: sharpness_score = 0
    return brightness_score, sharpness_score

# <<< MODIFIED >>>: Function updated to use preference logic for plate selection.
def recognize_plate(image_crop, api_key, track_id, csv_writer, frame_number, crop_filename):
    """Sends a cropped image to Plate Recognizer API and logs detailed results."""
    csv_data = {
        "frame": frame_number, "id": track_id, "plate": "N/A", "plate_score": 0.0,
        "filename": crop_filename.name, "api_guess": "", "vehicle_type": "",
        "vehicle_score": 0.0, "region_code": "", "region_score": 0.0, "error_msg": ""
    }

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        logging.warning(f"ID {track_id}: Skipping Plate Recognizer, API key not set.")
        csv_data["plate"] = "NO_API_KEY"
        csv_writer.writerow(csv_data.values())
        return

    logging.info(f"ID {track_id}: Sending crop to Plate Recognizer API...")
    plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$')
    api_url = "https://api.platerecognizer.com/v1/plate-reader/"; headers = {"Authorization": f"Token {api_key}"}

    try:
        _, buffer = cv2.imencode('.jpg', image_crop)
        response = requests.post(api_url, files=dict(upload=buffer.tobytes()), headers=headers, timeout=30)
        
        if response.status_code == 429:
            logging.warning(f"ID {track_id}: API rate limit hit. Waiting for 1 second and retrying.")
            time.sleep(1)
            response = requests.post(api_url, files=dict(upload=buffer.tobytes()), headers=headers, timeout=30)

        data = response.json()
        try: logging.info(f"ID {track_id}: Full API Response:\n{json.dumps(data, indent=2)}")
        except json.JSONDecodeError: logging.info(f"ID {track_id}: Raw API Response Text: {response.text}")
        
        if response.status_code not in [200, 201]:
            error_detail = data.get('detail', response.text)
            logging.error(f"ID {track_id}: API error {response.status_code}: {error_detail}")
            csv_data["plate"] = "API_ERROR"; csv_data["error_msg"] = f"Code {response.status_code}: {error_detail}"
            csv_writer.writerow(csv_data.values())
            return
        
        if not (data.get('results') and len(data['results']) > 0):
            logging.info(f"ID {track_id}: API returned no plate in the crop.")
            csv_data["plate"] = "NOT_FOUND"
            csv_writer.writerow(csv_data.values())
            return

        plate_data = data['results'][0]
        csv_data["api_guess"] = plate_data.get('plate', '')
        vehicle_info = plate_data.get('vehicle', {}); csv_data["vehicle_type"] = vehicle_info.get('type', ''); csv_data["vehicle_score"] = vehicle_info.get('score', 0.0)
        region_info = plate_data.get('region', {}); csv_data["region_code"] = region_info.get('code', ''); csv_data["region_score"] = region_info.get('score', 0.0)
        
        candidates = plate_data.get('candidates', [])
        valid_candidates = [c for c in candidates if plate_pattern.match(c.get('plate', '').upper().replace(' ', ''))]
        
        if valid_candidates:
            # Helper function to adjust score based on preferred state codes
            def get_adjusted_score(candidate):
                plate_str = candidate.get('plate', '').upper()
                original_score = candidate.get('score', 0.0)
                # If the plate starts with a preferred code, add a bonus
                if any(plate_str.startswith(code) for code in PREFERRED_STATE_CODES):
                    logging.debug(f"Applying bonus to {plate_str}. Original: {original_score:.3f}, New: {original_score + PREFERENCE_BONUS:.3f}")
                    return original_score + PREFERENCE_BONUS
                return original_score

            # Find the best candidate using the adjusted score
            best = max(valid_candidates, key=get_adjusted_score)
            
            # Save the original API score to the CSV, not the adjusted one
            csv_data["plate"] = best['plate']
            csv_data["plate_score"] = best['score'] # This is the original score
            logging.info(f"ID {track_id}: SUCCESS (Preference Match)! Plate: {csv_data['plate']}, Original Score: {csv_data['plate_score']:.2f}")

        else:
            csv_data["plate"] = "INVALID_FORMAT"
            csv_data["plate_score"] = 0.0
            logging.warning(f"ID {track_id}: No candidates matched the pattern. API Top Guess was: {csv_data['api_guess']}")

        csv_writer.writerow(csv_data.values())

    except requests.exceptions.Timeout:
        logging.error(f"ID {track_id}: API request timed out.")
        csv_data["plate"] = "API_TIMEOUT"; csv_data["error_msg"] = "Request timed out"
        csv_writer.writerow(csv_data.values())
    except Exception as e:
        logging.error(f"ID {track_id}: Exception during API call: {e}")
        csv_data["plate"] = "EXCEPTION"; csv_data["error_msg"] = str(e)
        csv_writer.writerow(csv_data.values())


def process_and_recognize_shot(vid, best_shot_data, crops_output_dir, csv_writer):
    if not best_shot_data:
        logging.warning(f"ID {vid}: No valid shot data found, cannot process.")
        return
    logging.info(f"ID {vid}: Processing best shot from frame {best_shot_data['frame_number']} with quality score {best_shot_data['score']:.2f}.")
    x1, y1, x2, y2 = map(int, best_shot_data['vehicle_box'])
    cropped_vehicle = best_shot_data['frame_data'][y1:y2, x1:x2]
    if cropped_vehicle.size > 0:
        crop_filename = crops_output_dir / f"vehicle_id_{vid}_best_shot_score_{int(best_shot_data['score'])}.jpg"
        cv2.imwrite(str(crop_filename), cropped_vehicle)
        recognize_plate(cropped_vehicle, PLATE_RECOGNIZER_API_KEY, vid, csv_writer, best_shot_data['frame_number'], crop_filename)
    else:
        logging.warning(f"ID {vid}: Failed to create a valid crop.")


def run_tracker_on_video(model, input_video_path, output_video_path, crops_output_dir, results_csv_path, confidence, vehicle_class_name, plate_class_name):
    class_names = model.names
    cap = cv2.VideoCapture(str(input_video_path))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    
    display_dims = None
    if RESIZE_PREVIEW_WINDOW:
        sw, sh = get_display_resolution()
        scale = min((sw * PREVIEW_WINDOW_SCALE) / frame_width, (sh * PREVIEW_WINDOW_SCALE) / frame_height)
        if scale < 1.0: display_dims = (int(frame_width * scale), int(frame_height * scale))
    WINDOW_NAME = "YOLOv8 Vehicle Tracking"; cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    best_vehicle_shots = {}

    vehicle_positions = {}
    vehicle_stationary_frames = {}
    global_inactive_frames = 0
    inactivity_threshold_frames = int(INACTIVITY_SECONDS * fps) if fps > 0 else (INACTIVITY_SECONDS * 30)
    logging.info(f"Motion detection enabled. Inactivity threshold: {inactivity_threshold_frames} frames ({INACTIVITY_SECONDS} seconds).")

    with open(results_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_header = [
            "Frame", "Vehicle_ID", "Recognized_Plate (Pattern Match)", "Plate_Confidence",
            "Crop_Filename", "API_Top_Guess (No Match)", "Vehicle_Type", "Vehicle_Type_Score",
            "Region_Code", "Region_Code_Score", "API_Error_Message"
        ]
        csv_writer.writerow(csv_header)
        
        logging.info("--- Starting Video Processing ---")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            vehicles = {}
            
            frame_plate_metrics = {}
            results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=confidence)
            
            current_vehicles_on_screen = set()
            all_vehicles_are_stationary = True

            if results[0].boxes.id is not None:
                all_boxes, track_ids, class_ids = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.int().cpu().tolist(), results[0].boxes.cls.int().cpu().tolist()
                
                vehicles = {tid: box for box, tid, cid in zip(all_boxes, track_ids, class_ids) if class_names[cid] == vehicle_class_name}
                plates = {tuple(map(int, box)): cid for box, cid in zip(all_boxes, class_ids) if class_names[cid] == plate_class_name}

                for vid, v_box in vehicles.items():
                    current_vehicles_on_screen.add(vid)
                    x1, y1, x2, y2 = v_box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    if vid in vehicle_positions:
                        prev_x, prev_y = vehicle_positions[vid]
                        distance = math.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        
                        if distance < STATIONARY_THRESHOLD_PIXELS:
                            vehicle_stationary_frames[vid] = vehicle_stationary_frames.get(vid, 0) + 1
                        else:
                            vehicle_stationary_frames[vid] = 0
                            all_vehicles_are_stationary = False
                    else:
                        vehicle_stationary_frames[vid] = 0
                        all_vehicles_are_stationary = False
                    
                    vehicle_positions[vid] = (center_x, center_y)

                    for p_box, cid in plates.items():
                        if do_boxes_intersect(v_box, p_box):
                            px1, py1, px2, py2 = map(int, p_box)
                            plate_crop = frame[py1:py2, px1:px2]
                            plate_area = get_box_area(p_box)
                            brightness, sharpness = calculate_plate_quality(plate_crop)
                            current_score = plate_area * (brightness ** BRIGHTNESS_WEIGHT) * (math.log1p(sharpness) ** SHARPNESS_WEIGHT)
                            
                            frame_plate_metrics[tuple(p_box)] = {'score': current_score, 'area': plate_area, 'brightness': brightness, 'sharpness': sharpness}

                            if current_score > best_vehicle_shots.get(vid, {}).get('score', -1):
                                logging.debug(f"ID {vid}: New best shot candidate. Score: {current_score:.2f}")
                                best_vehicle_shots[vid] = {'score': current_score, 'vehicle_box': v_box, 'frame_data': frame.copy(), 'frame_number': frame_count}
                            break
            else:
                 all_vehicles_are_stationary = True

            if not vehicles:
                 global_inactive_frames += 1
            elif all_vehicles_are_stationary:
                global_inactive_frames += 1
                logging.debug(f"All vehicles are stationary. Global inactivity counter: {global_inactive_frames}/{inactivity_threshold_frames}")
            else:
                global_inactive_frames = 0

            if global_inactive_frames > inactivity_threshold_frames:
                logging.info(f"All vehicles have been stationary for over {INACTIVITY_SECONDS} seconds. Stopping video processing.")
                break

            disappeared_ids = set(vehicle_positions.keys()) - current_vehicles_on_screen
            for vid in disappeared_ids:
                if vid in vehicle_positions: del vehicle_positions[vid]
                if vid in vehicle_stationary_frames: del vehicle_stationary_frames[vid]

            final_frame = frame.copy()
            if results[0].boxes.id is not None:
                for box, track_id, cls_id in zip(all_boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    if class_names[cls_id] == vehicle_class_name:
                        is_stationary = vehicle_stationary_frames.get(track_id, 0) > fps
                        color = (0, 0, 255) if is_stationary else (255, 0, 0)
                        label_text = f"ID:{track_id}" + (" (S)" if is_stationary else "")
                        cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 2)
                        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(final_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                        cv2.putText(final_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                    elif class_names[cls_id] == plate_class_name:
                        cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        box_key = (x1, y1, x2, y2)
                        if box_key in frame_plate_metrics:
                            metrics = frame_plate_metrics[box_key]
                            info_text = f"Scr: {metrics['score']:.0f} A: {int(metrics['area'])} B: {metrics['brightness']:.2f} S: {metrics['sharpness']:.0f}"
                            font_scale = 0.5; font_thickness = 1; padding = 5
                            (w, h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(final_frame, (x1, y1 - h - padding), (x1 + w, y1), (0, 255, 255), -1)
                            cv2.putText(final_frame, info_text, (x1, y1 - (padding // 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            out.write(final_frame)
            if RESIZE_PREVIEW_WINDOW and display_dims:
                frame_to_show = cv2.resize(final_frame, display_dims, interpolation=cv2.INTER_AREA)
            else: frame_to_show = final_frame
            cv2.imshow(WINDOW_NAME, frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            
        logging.info("--- Video processing loop finished. Now processing all best shots. ---")
        for vid, shot_data in best_vehicle_shots.items():
            logging.info(f"Processing final best shot for Vehicle ID: {vid}")
            process_and_recognize_shot(vid, shot_data, crops_output_dir, csv_writer)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("--- All operations complete. ---")

def main():
    logging.info("============================================="); logging.info("  YOLO Video Tracker (Motion-Detection Stop)"); logging.info("=============================================")
    model_path = Path(MODEL_PATH); input_video_path = Path(INPUT_VIDEO_PATH)
    if not model_path.exists(): logging.critical(f"FATAL: Model not found at '{MODEL_PATH}'"); return
    if not input_video_path.exists(): logging.critical(f"FATAL: Input video not found at '{INPUT_VIDEO_PATH}'"); return
    try:
        logging.info("Loading model..."); model = YOLO(model_path); class_names = list(model.names.values())
        logging.info(f"Model classes found: {class_names}"); vehicle_class_name, plate_class_name = None, None
        VEHICLE_KEYWORDS = ['vehicle', 'car', 'truck', 'bus', 'auto']; PLATE_KEYWORDS = ['plate', 'number', 'licence', 'license']
        for name in class_names:
            if any(k in name.lower() for k in VEHICLE_KEYWORDS): vehicle_class_name = name; logging.info(f"Auto-identified VEHICLE class: '{name}'")
            elif any(k in name.lower() for k in PLATE_KEYWORDS): plate_class_name = name; logging.info(f"Auto-identified PLATE class: '{name}'")
        if not vehicle_class_name or not plate_class_name: logging.critical("Could not auto-determine class names for 'vehicle' and 'plate'."); return
    except Exception as e: logging.critical(f"Failed to load model or detect class names: {e}"); return
    
    output_path_obj = Path(OUTPUT_PATH)
    output_video_path = output_path_obj / f"{input_video_path.stem}_tracked_final.mp4"
    crops_output_dir = output_path_obj / f"{input_video_path.stem}_crops"
    os.makedirs(crops_output_dir, exist_ok=True)
    results_csv_path = output_path_obj / f"{input_video_path.stem}_results.csv"
    
    run_tracker_on_video(model, input_video_path, output_video_path, crops_output_dir, results_csv_path, CONFIDENCE_THRESHOLD, vehicle_class_name, plate_class_name)
    logging.info(f"Tracked video saved to: {output_video_path}"); logging.info(f"Cropped images saved in: {crops_output_dir}"); logging.info(f"API results saved to: {results_csv_path}")


if __name__ == "__main__":
    main()
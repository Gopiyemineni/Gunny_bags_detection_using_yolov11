import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os

# Initialize tkinter for file dialog
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

# Prompt user to select a video file
print("Please select a video file for object detection.")
video_path = filedialog.askopenfilename(
    title="Select Video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("No video selected. Exiting.")
    exit()

# Load the YOLO model
model = YOLO(r"C:\Users\admin\Downloads\analysis code\gudivada_ch3_20250619152123_20250619172534_clip.pt")  # Replace with your model.pt path if not in the same directory

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer to save output
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Resize frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert BGR to RGB for YOLO model
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Perform object detection with tracking
    results = model.track(
        frame_rgb,
        persist=True,
        tracker="botsort.yaml",
        conf=0.2,
        iou=0.45,
        device="cpu",
        imgsz=640,
        verbose=False
    )

    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Convert back to BGR for OpenCV display and saving
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow("YOLO Full Frame Detection", annotated_frame_bgr)

    # Write the frame to the output video
    out.write(annotated_frame_bgr)

    # Press 'q' to quit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detection completed. Output video saved as {output_path}.")
print(f"Results directory: {output_dir}/")
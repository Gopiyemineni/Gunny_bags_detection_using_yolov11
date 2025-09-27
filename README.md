# **Gunny Bags Detection using YOLOv11**

This project is a real-time video analysis system that allows users to upload videos, define regions of interest (ROI) or reference lines, and analyze objects (e.g., gunny bags) using a YOLOv11 model. The system provides a web interface and uses Flask + Socket.IO for interactive video processing.

---

## 🎥 Demo Output

**Watch the processed result video below:**

[Output Video](https://github.com/<username>/<repo>/blob/push-output-video/output.mp4?raw=true)

---

## Features:
- Upload videos and display the first frame for annotation
- Define ROI or reference lines for analysis
- Real-time object detection and tracking using YOLOv11
- Toggle frame display during processing
- Background video processing for multiple clients simultaneously
- Saves analysis results in a dedicated results folder

## Requirements:
- Python 3.10+
- Flask
- Flask-SocketIO
- Eventlet
- OpenCV
- Base64 (Python standard library)
- YOLOv11 model weights (.pt files)

## Installation:
```bash
git clone <repo-url>
cd Gunny_bags_detection_using_yolov11
pip install -r requirements.txt
mkdir uploads results




## Usage

To run the Gunny Bags Detection system, follow these steps:

```bash
python app.py
Open your browser at: http://127.0.0.1:5000

1.Upload a video file.
2.Draw ROI or reference line (if needed) and start processing.
3.Processed results will be saved in the results folder.

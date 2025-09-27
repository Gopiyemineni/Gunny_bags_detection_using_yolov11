                                                                       Gunny Bags Detection using YOLOv11

This project is a real-time video analysis system that allows users to upload videos, define regions of interest (ROI) or reference lines, and analyze objects (e.g., gunny bags) using a YOLOv11 model. The system provides a web interface and uses Flask + Socket.IO for interactive video processing.

Features:
Upload videos and display the first frame for annotation.
Define ROI or reference lines for analysis.
Real-time object detection and tracking using YOLOv11.
Toggle frame display during processing.
Background video processing for multiple clients simultaneously.
Saves analysis results in a dedicated results folder.

Requirements:
Python 3.10+
Flask
Flask-SocketIO
Eventlet
OpenCV
Base64 (Python standard library)
YOLOv11 model weights (.pt files)

Installation:
Clone the repository:
git clone <repo-url>
cd Gunny_bags_detection_using_yolov11
Install required packages:
pip install -r requirements.txt


Ensure directories exist:
mkdir uploads results

Usage:
Run the Flask app:
python app.py
Open your browser at:
http://127.0.0.1:5000

Upload a video file.

Draw ROI or reference line (if needed) and start processing.

The processed results will be saved in the results folder.

File Structure
Gunny_bags_detection_using_yolov11/
│
├─ app.py                  # Main Flask application
├─ video_analyzer.py       # Video processing & YOLOv11 logic
├─ templates/
│   └─ index.html          # Web interface
├─ uploads/                # Uploaded videos
├─ results/                # Processed results
└─ requirements.txt        # Python dependencies

Notes:
Make sure your YOLOv11 model .pt files are present in the repository or tracked with Git LFS.
The app uses Eventlet for async processing.
Each client session is tracked individually for real-time updates.

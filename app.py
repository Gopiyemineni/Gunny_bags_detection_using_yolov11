 # app.py
import eventlet
eventlet.monkey_patch() # Must be at the very top

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room
import os
import cv2
import base64
from video_analyzer import VideoAnalyzer, analyzers_by_sid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, async_mode='eventlet')

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    filename = file.filename
    video_path = os.path.join('uploads', filename)
    file.save(video_path)

    # Get the first frame for the user to draw on
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Could not read video file'}), 500

    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    height, width, _ = frame.shape

    return jsonify({
        'message': 'Video uploaded successfully',
        'video_filename': filename,
        'first_frame': frame_b64,
        'width': width,
        'height': height
    })

@socketio.on('start_processing')
def handle_start_processing(data):
    video_filename = data['video_filename']
    video_path = os.path.join('uploads', video_filename)
    sid = request.sid  # Get the Socket.IO session id for this client

    config = {
        'video_path': video_path,
        'model_path': video_filename.replace('.mp4', '.pt'),
        'roi': data.get('roi'),
        'line': data['line'],
        'process_full': data['process_full'],
        'direction_arrow': data['direction_arrow'],
        'display_frames': data.get('display_frames', True)
    }

    print("Starting analysis with config:", config)
    join_room(sid)  # Put this client in their own room
    analyzer = VideoAnalyzer(config, socketio, sid)  # Pass sid to VideoAnalyzer
    socketio.start_background_task(analyzer.process_video)

@socketio.on('toggle_display_frames')
def handle_toggle_display_frames(data):
    sid = request.sid
    value = data.get('display_frames', True)
    analyzer = analyzers_by_sid.get(sid)
    if analyzer:
        analyzer.set_display_frames(value)
        print(f"[SID {sid}] Set display_frames to {value}")

if __name__ == '__main__':
    print("Server starting at http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
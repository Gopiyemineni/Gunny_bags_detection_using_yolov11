// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // UI Elements
    const videoUpload = document.getElementById('video-upload');
    const configArea = document.getElementById('config-area');
    const processingArea = document.getElementById('processing-area');
    const setupCanvas = document.getElementById('setup-canvas');
    const ctx = setupCanvas.getContext('2d');
    const startBtn = document.getElementById('start-btn');
    const resetBtn = document.getElementById('reset-drawings-btn');
    const countDisplay = document.getElementById('count-display');
    const videoStream = document.getElementById('video-stream');
    const statusDiv = document.getElementById('status');
    const processModeRadios = document.querySelectorAll('input[name="process-mode"]');

    // Add a checkbox for disabling frame display
    const displayFramesCheckbox = document.createElement('input');
    displayFramesCheckbox.type = 'checkbox';
    displayFramesCheckbox.id = 'display-frames-checkbox';
    displayFramesCheckbox.checked = true;
    const displayFramesLabel = document.createElement('label');
    displayFramesLabel.htmlFor = 'display-frames-checkbox';
    displayFramesLabel.textContent = ' Display video frames during processing (uses more RAM)';
    const controlsDiv = document.getElementById('config-area');
    controlsDiv.insertBefore(displayFramesCheckbox, controlsDiv.firstChild);
    controlsDiv.insertBefore(displayFramesLabel, displayFramesCheckbox.nextSibling);

    // State variables
    let firstFrame = new Image();
    let videoFilename = null;
    let isDrawing = false;
    let drawingMode = 'roi'; // 'roi' -> 'line' -> 'direction'
    let roi = null;
    let line = null;
    let directionArrow = null;
    let startPoint = null;
    let scaleFactor = 1;
    let originalWidth = 0, originalHeight = 0;

    // --- Step 1: Video Upload ---
    videoUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;

        statusDiv.textContent = 'Uploading video...';
        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            statusDiv.textContent = '';
            videoFilename = data.video_filename;
            firstFrame.src = `data:image/jpeg;base64,${data.first_frame}`;
            firstFrame.onload = () => {
                originalWidth = firstFrame.width;
                originalHeight = firstFrame.height;
                // Calculate scale factor to fit within 640x480
                const maxW = 640, maxH = 480;
                scaleFactor = Math.min(maxW / originalWidth, maxH / originalHeight, 1);
                setupCanvas.width = Math.round(originalWidth * scaleFactor);
                setupCanvas.height = Math.round(originalHeight * scaleFactor);
                configArea.classList.remove('hidden');
                resetDrawings();
            };
        })
        .catch(error => {
            statusDiv.textContent = `Error: ${error.message}`;
            console.error('Upload Error:', error);
        });
    });
    
    // --- Step 2: Configuration and Drawing ---
    processModeRadios.forEach(radio => {
        radio.addEventListener('change', resetDrawings);
    });

    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }
    
    function drawArrow(ctx, fromx, fromy, tox, toy) {
        const headlen = 15; // length of head in pixels
        const dx = tox - fromx;
        const dy = toy - fromy;
        const angle = Math.atan2(dy, dx);
        ctx.beginPath();
        ctx.moveTo(fromx, fromy);
        ctx.lineTo(tox, toy);
        ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(tox, toy);
        ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
    }

    function redrawCanvas() {
        ctx.clearRect(0, 0, setupCanvas.width, setupCanvas.height);
        ctx.drawImage(firstFrame, 0, 0, setupCanvas.width, setupCanvas.height);

        // Draw ROI
        if (roi) {
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
        }

        // Draw Line
        if (line) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(line.start.x, line.start.y);
            ctx.lineTo(line.end.x, line.end.y);
            ctx.stroke();
        }

        // Draw Direction Arrow
        if (directionArrow) {
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 3;
            drawArrow(ctx, directionArrow.start.x, directionArrow.start.y, directionArrow.end.x, directionArrow.end.y);
        }
        validateConfig();
    }
    
    function setDrawingMode() {
        const isRoiMode = document.querySelector('input[name="process-mode"]:checked').value === 'roi';
        if (isRoiMode && !roi) {
            drawingMode = 'roi';
        } else if (!line) {
            drawingMode = 'line';
        } else if (!directionArrow) {
            drawingMode = 'direction';
        } else {
            drawingMode = 'done';
        }
    }
    
    function resetDrawings() {
        roi = null;
        line = null;
        directionArrow = null;
        setDrawingMode();
        redrawCanvas();
    }
    
    resetBtn.addEventListener('click', resetDrawings);

    setupCanvas.addEventListener('mousedown', (e) => {
        if (drawingMode === 'done') return;
        startPoint = getMousePos(setupCanvas, e);
        isDrawing = true;
    });

    setupCanvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        redrawCanvas();
        const currentPoint = getMousePos(setupCanvas, e);
        
        ctx.beginPath();
        if (drawingMode === 'roi') {
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.strokeRect(startPoint.x, startPoint.y, currentPoint.x - startPoint.x, currentPoint.y - startPoint.y);
        } else if (drawingMode === 'line') {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
            ctx.moveTo(startPoint.x, startPoint.y);
            ctx.lineTo(currentPoint.x, currentPoint.y);
            ctx.stroke();
        } else if (drawingMode === 'direction') {
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 3;
            drawArrow(ctx, startPoint.x, startPoint.y, currentPoint.x, currentPoint.y);
        }
    });

    setupCanvas.addEventListener('mouseup', (e) => {
        if (!isDrawing) return;
        isDrawing = false;
        const endPoint = getMousePos(setupCanvas, e);

        if (drawingMode === 'roi') {
            roi = {
                x: Math.min(startPoint.x, endPoint.x),
                y: Math.min(startPoint.y, endPoint.y),
                w: Math.abs(startPoint.x - endPoint.x),
                h: Math.abs(startPoint.y - endPoint.y)
            };
        } else if (drawingMode === 'line') {
            line = { start: startPoint, end: endPoint };
        } else if (drawingMode === 'direction') {
            directionArrow = { start: startPoint, end: endPoint };
        }
        setDrawingMode();
        redrawCanvas();
    });

    function validateConfig() {
        const isRoiMode = document.querySelector('input[name="process-mode"]:checked').value === 'roi';
        let isReady = false;
        if (isRoiMode) {
             isReady = (roi && line && directionArrow);
        } else {
             isReady = (line && directionArrow);
        }
        startBtn.disabled = !isReady;
    }

    // --- Step 3: Start Processing ---
    startBtn.addEventListener('click', () => {
        const processFull = document.querySelector('input[name="process-mode"]:checked').value === 'full';

        function scaleUp(point) {
            return [Math.round(point.x / scaleFactor), Math.round(point.y / scaleFactor)];
        }
        function scaleUpRect(rect) {
            return [
                Math.round(rect.x / scaleFactor),
                Math.round(rect.y / scaleFactor),
                Math.round(rect.w / scaleFactor),
                Math.round(rect.h / scaleFactor)
            ];
        }

        const config = {
            video_filename: videoFilename,
            process_full: processFull,
            roi: roi ? scaleUpRect(roi) : null,
            line: [scaleUp(line.start), scaleUp(line.end)],
            direction_arrow: [scaleUp(directionArrow.start), scaleUp(directionArrow.end)],
            display_frames: displayFramesCheckbox.checked
        };
        
        socket.emit('start_processing', config);
        
        configArea.classList.add('hidden');
        processingArea.classList.remove('hidden');
        statusDiv.textContent = 'Processing... This may take a while.';
    });

    // --- Socket.IO Event Handlers ---
    socket.on('update_frame', (data) => {
        const blob = new Blob([data.image], { type: 'image/jpeg' });
        videoStream.src = URL.createObjectURL(blob);
    });
    socket.on('count_update', (data) => {
        countDisplay.textContent = `Count: ${data.count}`;
    });
    socket.on('processing_complete', (data) => {
        statusDiv.innerHTML = `Processing complete! Final count: ${data.count}. <br> \
                               Report saved to: <strong>${data.json_path}</strong> <br>\
                               Processed video saved to: <strong>${data.video_path}</strong>`;
    });
    socket.on('processing_error', (data) => {
        statusDiv.textContent = `Error: ${data.msg}`;
    });
    socket.on('status_update', (data) => {
        statusDiv.textContent = data.msg;
    });

    displayFramesCheckbox.addEventListener('change', () => {
        // If processing area is visible, emit toggle event
        if (!processingArea.classList.contains('hidden')) {
            socket.emit('toggle_display_frames', { display_frames: displayFramesCheckbox.checked });
        }
    });
});
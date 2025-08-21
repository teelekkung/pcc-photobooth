import gphoto2 as gp
import cv2
import numpy as np
import time
import threading
import io
import os
import signal
import sys
from datetime import datetime
from flask import Flask, Response, render_template, request, send_file

app = Flask(__name__)

# Directory to save captured images
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

camera = None
camera_name = None
latest_frame = None
captured_image = None
captured_filename = None
running = True
lock = threading.Lock()
mode = "live"  # "live", "capture", "captured"


def list_cameras():
    """Return a list of available cameras."""
    cameras = gp.Camera.autodetect()
    return cameras  # list of (port, model)


def connect_camera(camera_port=None):
    """Connect to a camera. If port is None, connect to first available camera."""
    cam = gp.Camera()
    try:
        if camera_port:
            cam.init(gp.Context())
        else:
            cam.init()
        summary = cam.get_summary()
        print(f"[INFO] Connected to camera:\n{str(summary)}")
    except gp.GPhoto2Error as e:
        print(f"[ERROR] Camera init failed: {e}")
        cam = None
    return cam


def capture_loop():
    """Background thread to grab frames or full-resolution image."""
    global latest_frame, captured_image, captured_filename, camera, running, mode

    cam = connect_camera(camera_name)

    while running:
        try:
            if mode == "live" and cam:
                # Low-res preview for streaming
                camera_file = cam.capture_preview()
                file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
                data = memoryview(file_data).tobytes()
                np_arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is not None:
                    img = cv2.flip(img, 1)  # mirror horizontally
                    ret, buffer = cv2.imencode('.jpg', img)
                    with lock:
                        latest_frame = buffer.tobytes()

            elif mode == "capture" and cam:
                # Full-resolution capture
                file_path = cam.capture(gp.GP_CAPTURE_IMAGE)
                camera_file = cam.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
                data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
                img_bytes = memoryview(data).tobytes()

                # Save to server with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                with lock:
                    captured_image = img_bytes
                    captured_filename = filepath
                    latest_frame = captured_image  # show captured image in feed
                    mode = "captured"

                # Optionally delete from camera after capture
                cam.file_delete(file_path.folder, file_path.name)

            else:
                time.sleep(0.1)

        except gp.GPhoto2Error as e:
            print(f"[WARN] Camera error: {e}")
            try:
                cam.exit()
            except gp.GPhoto2Error:
                pass
            cam = None
            while running and cam is None:
                try:
                    cam = connect_camera(camera_name)
                    print("[INFO] Camera reconnected.")
                except gp.GPhoto2Error as e2:
                    print(f"[WARN] Reconnect failed: {e2}")
                    time.sleep(2)


def generate_frames():
    """Stream frames to clients."""
    global latest_frame
    while True:
        with lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

def cleanup(sig, frame):
    """Graceful shutdown on CTRL+C."""
    global running, camera
    print("\n[INFO] Shutting down server...")
    running = False
    # Exit camera safely
    try:
        if camera:
            print("nooooo")
            camera.exit()
    except:
        pass
    sys.exit(0)

@app.route('/')
def index():
    cameras = list_cameras()
    return render_template("index.html", cameras=cameras)


@app.route('/set_camera', methods=['POST'])
def set_camera():
    """Switch camera."""
    global camera_name
    selected_port = request.form.get('camera_port')
    print(f"[INFO] Switching camera to {selected_port}")
    camera_name = selected_port
    threading.Thread(target=capture_loop, daemon=True).start()
    return "OK", 200


@app.route('/capture', methods=['POST'])
def capture():
    """Trigger full-resolution capture."""
    global mode
    mode = "capture"
    return "Capturing", 200


@app.route('/return_live', methods=['POST'])
def return_live():
    """Return to live preview."""
    global mode
    mode = "live"
    return "Live", 200


@app.route('/download')
def download_image():
    """Download the last captured image."""
    global captured_image, captured_filename
    if captured_image and captured_filename:
        return send_file(captured_filename,
                         mimetype='image/jpeg',
                         as_attachment=True,
                         download_name=os.path.basename(captured_filename))
    else:
        return "No image captured", 404


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

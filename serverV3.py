#!/usr/bin/env python3
# serverV2.py — DSLR control on Raspberry Pi (Flask + gphoto2 + OpenCV)
# On-demand live: start preview only when /video_feed has viewers.

import os
import sys
import time
import signal
import threading
import mimetypes
from datetime import datetime

import cv2
import numpy as np
import gphoto2 as gp
from flask import Flask, Response, request, send_file, send_from_directory, jsonify
from flask_cors import CORS

# ------------------------------------------------------------
# Flask app & CORS
# ------------------------------------------------------------
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# Global state
# ------------------------------------------------------------
selected_port = None          # e.g. "usb:001,010"
latest_frame = None           # bytes (JPEG; for MJPEG stream)
captured_image = None         # bytes (for /download)
captured_filename = None      # str (path on disk)
mode = "live"                 # "live" or "captured" (informational)
running = False               # capture thread control
lock = threading.Lock()
capture_thread = None
supports_preview = None       # whether capture_preview works
viewers = 0                   # number of active /video_feed clients

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
EXT_MAP = {
    'image/jpeg': '.jpg',
    'image/x-canon-cr2': '.cr2',
    'image/x-nikon-nef': '.nef',
    'image/tiff': '.tif',
}

def _nocache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def _choose_ext(mime: str, fallback_name: str) -> str:
    ext = EXT_MAP.get(mime)
    if not ext:
        guessed = mimetypes.guess_extension(mime or '')
        ext = guessed or os.path.splitext(fallback_name)[1] or '.bin'
    return ext

def list_cameras():
    """Returns [(model, port), ...]"""
    return gp.Camera.autodetect()

def connect_camera(camera_port=None):
    """Init gphoto2 camera. If camera_port provided, bind to it."""
    cam = gp.Camera()
    try:
        if camera_port:
            pil = gp.PortInfoList()
            pil.load()
            idx = pil.lookup_path(camera_port)
            if idx < 0:
                raise gp.GPhoto2Error(gp.GP_ERROR_BAD_PARAMETERS)
            cam.set_port_info(pil[idx])

        cam.init()

        # Best-effort set to JPEG if supported (camera may ignore)
        try_set_image_jpeg(cam)

        try:
            summary = cam.get_summary()
            print(f"[INFO] Connected at {camera_port or '(auto)'}:\n{str(summary)}")
        except gp.GPhoto2Error:
            print(f"[INFO] Connected at {camera_port or '(auto)'} (no summary)")
        return cam
    except gp.GPhoto2Error as e:
        print(f"[ERROR] Camera init failed on port {camera_port}: {e}")
        try:
            cam.exit()
        except Exception:
            pass
        return None

def _check_preview_support(cam):
    """Test whether capture_preview works (camera live view)."""
    try:
        _ = cam.capture_preview()
        return True
    except gp.GPhoto2Error as e:
        print(f"[WARN] capture_preview not supported or not in live view: {e}")
        return False

def try_set_image_jpeg(cam):
    """Best-effort set image format to a JPEG mode (may be ignored)."""
    try:
        cfg = cam.get_config()
        for key in ('imageformat', 'imagequality'):
            try:
                node = cfg.get_child_by_name(key)
            except gp.GPhoto2Error:
                node = None
            if not node:
                continue
            choices = [node.get_choice(i) for i in range(node.count_choices())]
            for c in choices:
                lc = c.lower()
                if 'jpeg' in lc or 'jpg' in lc or 'fine' in lc:
                    node.set_value(c)
                    cam.set_config(cfg)
                    print(f"[INFO] Set {key} -> {c}")
                    return
    except gp.GPhoto2Error as e:
        print(f"[WARN] Cannot set JPEG via gphoto2: {e}")

def _safe_save_camera_file(camera_file, filepath: str):
    """Use gphoto2 to save the file to disk, then sanity-check if JPEG."""
    camera_file.save(filepath)
    if filepath.lower().endswith('.jpg'):
        with open(filepath, 'rb') as f:
            head = f.read(2)
        if head != b'\xff\xd8':
            raise ValueError("Saved file doesn't start with JPEG SOI marker")

def _update_latest_with_camera_preview(cam, cam_folder: str, cam_name: str):
    """
    Fetches camera-side preview JPEG for the captured file (works for RAW).
    Updates global latest_frame if successful.
    """
    global latest_frame, lock
    try:
        thumb = cam.file_get(cam_folder, cam_name, gp.GP_FILE_TYPE_PREVIEW)
        data = gp.check_result(gp.gp_file_get_data_and_size(thumb))
        thumb_bytes = memoryview(data).tobytes()
        if thumb_bytes[:2] == b'\xff\xd8':
            with lock:
                latest_frame = thumb_bytes
            return True
    except gp.GPhoto2Error:
        pass
    return False

def safe_capture_one(cam):
    """
    Captures one image, saves to disk with correct extension, and returns:
    (host_filepath, mime, cam_folder, cam_name)
    """
    file_path = cam.capture(gp.GP_CAPTURE_IMAGE)  # -> gp.CameraFilePath
    cam_folder, cam_name = file_path.folder, file_path.name
    camera_file = cam.file_get(cam_folder, cam_name, gp.GP_FILE_TYPE_NORMAL)

    mime = camera_file.get_mime_type()  # 'image/jpeg', 'image/x-nikon-nef', ...
    ext = _choose_ext(mime, cam_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    host_filename = f"capture_{ts}{ext}"
    host_filepath = os.path.join(SAVE_DIR, host_filename)

    _safe_save_camera_file(camera_file, host_filepath)

    # Optionally delete from camera (keeps SD card tidy)
    try:
        cam.file_delete(cam_folder, cam_name)
    except gp.GPhoto2Error:
        pass

    return host_filepath, mime, cam_folder, cam_name

# ------------------------------------------------------------
# Live capture thread (preview loop)
# ------------------------------------------------------------
def capture_loop():
    """Continuously update latest_frame from capture_preview when in live mode."""
    global latest_frame, mode, running, supports_preview

    cam = connect_camera(selected_port)
    if cam and supports_preview is None:
        supports_preview = _check_preview_support(cam)

    while running:
        try:
            if not cam:
                time.sleep(0.4)
                cam = connect_camera(selected_port)
                if cam and supports_preview is None:
                    supports_preview = _check_preview_support(cam)
                continue

            if mode == "live" and supports_preview:
                try:
                    camera_file = cam.capture_preview()
                    file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
                    data = memoryview(file_data).tobytes()
                    np_arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        ret, buffer = cv2.imencode('.jpg', img)
                        if ret:
                            with lock:
                                latest_frame = buffer.tobytes()
                except gp.GPhoto2Error as e:
                    print(f"[WARN] live preview failed: {e}")
                    supports_preview = False
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

        except gp.GPhoto2Error as e:
            print(f"[WARN] Camera error, reconnecting: {e}")
            try:
                cam.exit()
            except gp.GPhoto2Error:
                pass
            cam = None
            time.sleep(0.8)

    try:
        if cam:
            cam.exit()
    except Exception:
        pass
    print("[INFO] capture_loop stopped")

def generate_frames():
    """Yield latest_frame as multipart/x-mixed-replace JPEG stream."""
    global latest_frame
    while True:
        with lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)

def stop_capture_thread():
    global running, capture_thread
    if capture_thread and capture_thread.is_alive():
        running = False
        capture_thread.join(timeout=2)
    capture_thread = None

def start_capture_thread():
    global running, capture_thread, supports_preview, mode
    if running:
        return  # idempotent
    supports_preview = None
    mode = "live"
    running = True
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

def has_viewers():
    return viewers > 0

def ensure_preview_if_needed():
    """Start/stop live preview depending on active viewers."""
    if has_viewers():
        start_capture_thread()
    else:
        stop_capture_thread()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route('/')
def root():
    return "OK", 200

@app.route('/api/health')
def api_health():
    return jsonify({
        "ok": True,
        "running": running,
        "viewers": viewers,
        "mode": mode,
    }), 200

@app.route('/set_camera', methods=['POST'])
def set_camera():
    """
    Select a specific camera port.
    POST form: camera_port=usb:001,010
    """
    global selected_port, supports_preview, mode
    selected_port = request.form.get('camera_port')
    print(f"[INFO] Switching camera to port {selected_port}")
    supports_preview = None
    mode = "live"
    stop_capture_thread()
    ensure_preview_if_needed()  # (เดิมเรียก start_capture_thread() ตลอด)
    return "OK", 200

@app.route('/capture', methods=['POST'])
def capture():
    """
    Shoots one frame, saves with the correct extension, and updates the MJPEG stream
    ONLY with real JPEG bytes. If the capture is RAW, it tries the embedded preview.
    Returns JSON: {"url": "/captured_images/<filename>", "serverPath": "<abs path>"}
    """
    global mode, captured_image, captured_filename, selected_port, latest_frame

    # Pause preview so we don't fight for camera
    stop_capture_thread()
    cam = connect_camera(selected_port)
    if not cam:
        ensure_preview_if_needed()
        return "Camera not available", 503

    try:
        host_filepath, mime, cam_folder, cam_name = safe_capture_one(cam)

        # Keep bytes and filename for /download
        with open(host_filepath, 'rb') as f:
            captured_image = f.read()
        captured_filename = host_filepath

        # Update preview frame: prefer true JPEG; else try camera preview
        pushed = False
        if mime == 'image/jpeg' or host_filepath.lower().endswith('.jpg'):
            if captured_image[:2] == b'\xff\xd8':
                with lock:
                    latest_frame = captured_image
                pushed = True

        if not pushed:
            # For RAW captures, try a camera-generated preview JPEG
            _update_latest_with_camera_preview(cam, cam_folder, cam_name)

        mode = "captured"
        rel_url = f"/{os.path.join(SAVE_DIR, os.path.basename(host_filepath))}"
        return jsonify({"url": rel_url, "serverPath": host_filepath}), 200

    except Exception as e:
        print(f"[ERROR] capture failed: {e}")
        return f"Capture failed: {e}", 500
    finally:
        try:
            cam.exit()
        except Exception:
            pass
        # Clear the last still so the next /video_feed frame is fresh
        with lock:
            if mode == "live":
                latest_frame = None
        # resume preview only if there are viewers
        ensure_preview_if_needed()  # (เดิม start_capture_thread() เสมอ)

@app.route('/confirm', methods=['POST'])
def confirm():
    """
    Called after the user confirms the captured photo on the frontend.
    Clears captured state; return to live if there are viewers.
    Returns the live stream URL with a cache-buster.
    """
    global mode, captured_image, captured_filename, latest_frame
    captured_image = None
    captured_filename = None
    mode = "live"

    stop_capture_thread()
    with lock:
        latest_frame = None
    ensure_preview_if_needed()  # (เดิม start_capture_thread() เสมอ)

    ts = int(time.time() * 1000)
    return jsonify({"video": f"/video_feed?ts={ts}"}), 200

@app.route('/return_live', methods=['POST'])
def return_live():
    """
    Legacy route: same idea as /confirm but without returning a URL.
    """
    global mode, captured_image, captured_filename, latest_frame
    captured_image = None
    captured_filename = None
    mode = "live"
    stop_capture_thread()
    with lock:
        latest_frame = None
    ensure_preview_if_needed()  # (เดิม start_capture_thread() เสมอ)
    return "Live", 200

@app.route('/video_feed')
def video_feed():
    """Stream the live preview frames with viewer counting (on-demand start/stop)."""
    global viewers
    viewers += 1
    ensure_preview_if_needed()

    def stream():
        global viewers
        try:
            for chunk in generate_frames():
                yield chunk
        finally:
            viewers = max(0, viewers - 1)
            ensure_preview_if_needed()

    resp = Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers["X-Accel-Buffering"] = "no"
    return _nocache_headers(resp)

@app.route('/captured_images/<path:filename>')
def serve_captured_image(filename):
    resp = send_from_directory(SAVE_DIR, filename)
    return _nocache_headers(resp)

@app.route('/download')
def download_image():
    global captured_image, captured_filename
    if captured_image and captured_filename:
        resp = send_file(
            captured_filename,
            mimetype='image/jpeg' if captured_filename.lower().endswith('.jpg') else None,
            as_attachment=False,  # inline; change to True if you want download
            download_name=os.path.basename(captured_filename)
        )
        return _nocache_headers(resp)
    else:
        return "No image captured", 404

# ------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------
def cleanup(sig, frame):
    print("\n[INFO] Shutting down server...")
    stop_capture_thread()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':
    # On-demand: don't start preview until someone opens /video_feed
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

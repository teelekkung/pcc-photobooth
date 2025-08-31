import gphoto2 as gp
from flask_cors import CORS
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
CORS(app)

SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- State ---
selected_port = None           # path ของพอร์ตกล้องเช่น "usb:001,010"
latest_frame = None
captured_image = None
captured_filename = None
mode = "live"                  # "live", "capture", "captured"
running = False
lock = threading.Lock()
capture_thread = None
supports_preview = None        # จะตรวจครั้งแรกแล้วจำค่า

def list_cameras():
    # คืน [(model, port), ...]
    return gp.Camera.autodetect()

def connect_camera(camera_port=None):
    cam = gp.Camera()
    try:
        if camera_port:
            # map พอร์ตให้กับกล้อง
            pil = gp.PortInfoList()
            pil.load()
            idx = pil.lookup_path(camera_port)
            if idx < 0:
                raise gp.GPhoto2Error(gp.GP_ERROR_BAD_PARAMETERS)
            cam.set_port_info(pil[idx])
        cam.init()
        # log summary
        try:
            summary = cam.get_summary()
            print(f"[INFO] Connected to camera at {camera_port or '(auto)'}:\n{str(summary)}")
        except gp.GPhoto2Error:
            print(f"[INFO] Connected to camera at {camera_port or '(auto)'} (no summary)")
        return cam
    except gp.GPhoto2Error as e:
        print(f"[ERROR] Camera init failed on port {camera_port}: {e}")
        try:
            cam.exit()
        except Exception:
            pass
        return None

def _check_preview_support(cam):
    # พยายาม capture_preview หนึ่งครั้งเพื่อดูว่ารองรับไหม
    try:
        cf = cam.capture_preview()
        # ได้ก็ถือว่ารองรับ
        return True
    except gp.GPhoto2Error as e:
        print(f"[WARN] capture_preview not supported or camera not in live view: {e}")
        return False

def capture_loop():
    global latest_frame, captured_image, captured_filename, mode, running, supports_preview

    cam = connect_camera(selected_port)
    if cam and supports_preview is None:
        supports_preview = _check_preview_support(cam)

    while running:
        try:
            if not cam:
                time.sleep(0.5)
                cam = connect_camera(selected_port)
                if cam and supports_preview is None:
                    supports_preview = _check_preview_support(cam)
                continue

            if mode == "live":
                if supports_preview:
                    try:
                        camera_file = cam.capture_preview()
                        file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
                        data = memoryview(file_data).tobytes()
                        np_arr = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            # ปรับให้เหมาะกับเว็บแสดงผล
                            ret, buffer = cv2.imencode('.jpg', img)
                            if ret:
                                with lock:
                                    latest_frame = buffer.tobytes()
                    except gp.GPhoto2Error as e:
                        # หาก preview ล้มเหลว เป็นครั้งแรก -> ปิดรองรับ
                        print(f"[WARN] live preview failed this cycle: {e}")
                        supports_preview = False
                        time.sleep(0.1)
                else:
                    # ไม่มี preview: แสดงเฟรมล่าสุดค้างไว้
                    time.sleep(0.1)

            elif mode == "capture":
                file_path = cam.capture(gp.GP_CAPTURE_IMAGE)
                camera_file = cam.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
                data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
                img_bytes = memoryview(data).tobytes()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                with lock:
                    captured_image = img_bytes
                    captured_filename = filepath
                    latest_frame = captured_image
                    mode = "captured"

                # ลบไฟล์ในกล้อง (ถ้ากล้องอนุญาต)
                try:
                    cam.file_delete(file_path.folder, file_path.name)
                except gp.GPhoto2Error:
                    pass

            else:
                time.sleep(0.05)

        except gp.GPhoto2Error as e:
            print(f"[WARN] Camera error, will try reconnect: {e}")
            try:
                cam.exit()
            except gp.GPhoto2Error:
                pass
            cam = None
            time.sleep(1)

    # ออกจากลูป
    try:
        if cam:
            cam.exit()
    except Exception:
        pass
    print("[INFO] capture_loop stopped")

def generate_frames():
    global latest_frame
    while True:
        with lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

def stop_capture_thread():
    global running, capture_thread
    if capture_thread and capture_thread.is_alive():
        running = False
        capture_thread.join(timeout=2)
    capture_thread = None

def start_capture_thread():
    global running, capture_thread
    running = True
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

def cleanup(sig, frame):
    print("\n[INFO] Shutting down server...")
    stop_capture_thread()
    sys.exit(0)



@app.route('/set_camera', methods=['POST'])
def set_camera():
    global selected_port, supports_preview, mode
    selected_port = request.form.get('camera_port')  # เช่น usb:001,010
    print(f"[INFO] Switching camera to port {selected_port}")
    supports_preview = None     # reset ให้ตรวจใหม่
    mode = "live"
    stop_capture_thread()
    start_capture_thread()
    return "OK", 200

@app.route('/capture', methods=['POST'])
def capture():
    global mode
    mode = "capture"
    return "Capturing", 200

@app.route('/return_live', methods=['POST'])
def return_live():
    global mode
    mode = "live"
    return "Live", 200

@app.route('/download')
def download_image():
    global captured_image, captured_filename
    if captured_image and captured_filename:
        return send_file(captured_filename,
                         mimetype='image/jpeg',
                         as_attachment=True,
                         download_name=os.path.basename(captured_filename))
    else:
        return "No image captured", 404



signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == '__main__':
    # เริ่มโดยยังไม่เลือกพอร์ต ให้ผู้ใช้เลือกที่หน้า /
    start_capture_thread()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
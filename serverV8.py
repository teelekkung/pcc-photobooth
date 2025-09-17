#!/usr/bin/env python3
# serverV7.py — DSLR/Mirrorless control on Raspberry Pi (Flask + gphoto2)
# - Live preview (MJPEG) แบบจำกัด FPS และส่งเฉพาะ "เฟรมใหม่"
# - Auto-focus เมื่อเริ่ม Live View และก่อนถ่าย
# - รองรับ mirrorless: พยายามเปิด liveview/EVF หลายคีย์ (viewfinder/eosviewfinder/liveview ฯลฯ)
# - API: /video_feed /capture /confirm /return_live /cameras /set_camera /api/health /api/diag
# - เพิ่ม /stop_stream และ /stop ให้ตรงกับ UI
#
# ENV:
#   FRONTEND_ORIGIN=http://<ui-host>:3000
#   CAMERA_PORT=usb:001,010
#   PREVIEW_FPS=18   # (ปรับได้ 1–60)
#
# pip install flask flask-cors gphoto2

import os, sys, time, signal, threading, mimetypes
from datetime import datetime
import gphoto2 as gp
from flask import Flask, Response, request, send_file, send_from_directory, jsonify
from flask_cors import CORS

# ---------- ENV ----------
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
CAMERA_PORT_ENV = os.getenv("CAMERA_PORT", "").strip() or None

def _clamp_fps(x):
    try:
        v = float(x)
        return max(1.0, min(60.0, v))
    except Exception:
        return 18.0

PREVIEW_FPS_ENV = (os.getenv("PREVIEW_FPS") or "").strip()
preview_fps = _clamp_fps(PREVIEW_FPS_ENV or 18.0)

# ---------- APP ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "captured_images")
os.makedirs(SAVE_DIR, exist_ok=True)

selected_port = CAMERA_PORT_ENV
latest_frame = None
latest_frame_ver = 0      # ใช้ตรวจว่ามีเฟรมใหม่จริง ๆ
latest_frame_ts = 0.0
captured_image = None
captured_filename = None
mode = "live"
running = False
lock = threading.Lock()
capture_thread = None
supports_preview = None
viewers = 0
last_error = None

EXT_MAP = {
    'image/jpeg': '.jpg',
    'image/x-canon-cr2': '.cr2',
    'image/x-nikon-nef': '.nef',
    'image/tiff': '.tif',
}

# ---------- Utils ----------
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

def _set_latest_frame(data: bytes):
    """อัปเดตเฟรมล่าสุด + เพิ่มเวอร์ชัน เพื่อให้ฝั่งสตรีมรู้ว่าเฟรมเปลี่ยนแล้ว"""
    global latest_frame, latest_frame_ver, latest_frame_ts, lock
    with lock:
        latest_frame = data
        latest_frame_ver += 1
        latest_frame_ts = time.monotonic()

# ---------- Camera connect ----------
def list_cameras():
    try:
        arr = gp.Camera.autodetect()
        return arr or []
    except gp.GPhoto2Error as e:
        print(f"[ERROR] autodetect failed: {e}")
        return []

def _set_port(cam, camera_port):
    pil = gp.PortInfoList(); pil.load()
    idx = pil.lookup_path(camera_port)
    if idx < 0:
        raise gp.GPhoto2Error(gp.GP_ERROR_BAD_PARAMETERS)
    cam.set_port_info(pil[idx])

def connect_camera(camera_port=None):
    global last_error
    last_error = None
    cam = gp.Camera()
    port_to_use = camera_port
    if not port_to_use:
        detected = list_cameras()
        if detected:
            port_to_use = detected[0][1]
            print(f"[INFO] Auto-select port: {port_to_use}")
        else:
            last_error = "No camera detected by libgphoto2"
            print(f"[ERROR] {last_error}")
            return None
    try:
        _set_port(cam, port_to_use)
        cam.init()
        try_set_image_jpeg(cam)
        try_enable_liveview(cam)  # สำคัญสำหรับ mirrorless
        try:
            summary = cam.get_summary()
            print(f"[INFO] Connected at {port_to_use}:\n{str(summary)}")
        except gp.GPhoto2Error:
            print(f"[INFO] Connected at {port_to_use} (no summary)")
        return cam
    except gp.GPhoto2Error as e:
        last_error = f"Camera init failed on port {port_to_use}: {e}"
        print(f"[ERROR] {last_error}")
        try:
            cam.exit()
        except Exception:
            pass
        return None

# ---------- Config helpers ----------
def _cfg(cam):
    try:
        return cam.get_config()
    except gp.GPhoto2Error:
        return None

def _find_child(cfg, names):
    if not cfg:
        return None, None
    for name in names:
        try:
            node = cfg.get_child_by_name(name)
            if node:
                return cfg, node
        except gp.GPhoto2Error:
            pass
    return cfg, None

def _set_value(cam, cfg, node, value):
    try:
        node.set_value(value)
        cam.set_config(cfg)
        return True
    except gp.GPhoto2Error as e:
        print(f"[WARN] set_value({value}) failed for '{node.get_name() if node else '?'}': {e}")
        return False

# ---------- Mirrorless/liveview ----------
def try_enable_liveview(cam):
    cfg = _cfg(cam)
    if not cfg:
        return False
    lv_keys = ['viewfinder', 'liveview', 'eosviewfinder', 'movie', 'uilock']
    enabled = False
    for key in lv_keys:
        cfg, node = _find_child(cfg, [key])
        if node and node.get_type() in (gp.GP_WIDGET_TOGGLE, gp.GP_WIDGET_RADIO):
            truthy = 1 if node.get_type() == gp.GP_WIDGET_TOGGLE else None
            if truthy is None:
                for i in range(node.count_choices()):
                    c = node.get_choice(i).lower()
                    if any(k in c for k in ['on', 'live', 'enable', 'movie', 'viewfinder']):
                        truthy = node.get_choice(i); break
            if truthy is not None and _set_value(cam, cfg, node, truthy):
                enabled = True
    # แตะ capturetarget (บางรุ่นช่วยให้ preview/capture อยู่ร่วมกัน)
    cfg = _cfg(cam)
    if cfg:
        cfg, node = _find_child(cfg, ['capturetarget'])
        if node and node.get_type() == gp.GP_WIDGET_RADIO:
            try:
                _set_value(cam, cfg, node, node.get_value())
            except Exception:
                pass
    return enabled

def _check_preview_support(cam):
    try:
        _ = cam.capture_preview()
        return True
    except gp.GPhoto2Error as e:
        print(f"[WARN] capture_preview not supported or not in live view: {e}")
        return False

# ---------- Autofocus ----------
def try_autofocus(cam, timeout_s=0.6):
    cfg = _cfg(cam)
    if not cfg:
        return False
    tried = False
    ok = False
    cfg = _cfg(cam); cfg, node = _find_child(cfg, ['autofocusdrive', 'autofocus', 'triggerfocus'])
    if node:
        tried = True
        val = 1 if node.get_type() == gp.GP_WIDGET_TOGGLE else 'On'
        ok = _set_value(cam, cfg, node, val) or ok
    cfg = _cfg(cam); cfg, node = _find_child(cfg, ['eosremoterelease'])
    if node and node.get_type() == gp.GP_WIDGET_RADIO:
        tried = True
        half = release = None
        for i in range(node.count_choices()):
            c = node.get_choice(i).lower()
            if 'press half' in c or 'half' in c: half = node.get_choice(i)
            if 'release' in c: release = node.get_choice(i)
        if half:
            ok = _set_value(cam, cfg, node, half) or ok
            time.sleep(0.2)
            if release: _set_value(cam, cfg, node, release)
    cfg = _cfg(cam); cfg, node = _find_child(cfg, ['manualfocusdrive'])
    if node and node.get_type() in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
        tried = True
        choice = node.get_choice(0)
        for i in range(node.count_choices()):
            c = node.get_choice(i).lower()
            if 'near' in c or c == '1' or 'small' in c:
                choice = node.get_choice(i); break
        ok = _set_value(cam, cfg, node, choice) or ok
    if tried:
        time.sleep(min(max(timeout_s, 0.1), 2.0))
    return ok

def try_set_image_jpeg(cam):
    try:
        cfg = cam.get_config()
        for key in ('imageformat', 'imagequality'):
            try:
                node = cfg.get_child_by_name(key)
            except gp.GPhoto2Error:
                node = None
            if not node: continue
            for i in range(node.count_choices()):
                c = node.get_choice(i)
                lc = c.lower()
                if 'jpeg' in lc or 'jpg' in lc or 'fine' in lc:
                    node.set_value(c); cam.set_config(cfg)
                    print(f"[INFO] Set {key} -> {c}")
                    return
    except gp.GPhoto2Error as e:
        print(f"[WARN] Cannot set JPEG via gphoto2: {e}")

def _safe_save_camera_file(camera_file, filepath: str):
    camera_file.save(filepath)
    if filepath.lower().endswith('.jpg'):
        with open(filepath, 'rb') as f:
            if f.read(2) != b'\xff\xd8':
                raise ValueError("Saved file doesn't start with JPEG SOI marker")

def _update_latest_with_camera_preview(cam, cam_folder: str, cam_name: str):
    try:
        thumb = cam.file_get(cam_folder, cam_name, gp.GP_FILE_TYPE_PREVIEW)
        data = gp.check_result(gp.gp_file_get_data_and_size(thumb))
        b = memoryview(data).tobytes()
        if b[:2] == b'\xff\xd8':
            _set_latest_frame(b)
            return True
    except gp.GPhoto2Error:
        pass
    return False

def safe_capture_one(cam):
    file_path = cam.capture(gp.GP_CAPTURE_IMAGE)
    cam_folder, cam_name = file_path.folder, file_path.name
    camera_file = cam.file_get(cam_folder, cam_name, gp.GP_FILE_TYPE_NORMAL)
    mime = camera_file.get_mime_type()
    ext = _choose_ext(mime, cam_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    host_filename = f"capture_{ts}{ext}"
    host_filepath = os.path.join(SAVE_DIR, host_filename)
    _safe_save_camera_file(camera_file, host_filepath)
    try:
        cam.file_delete(cam_folder, cam_name)
    except gp.GPhoto2Error:
        pass
    return host_filepath, mime, cam_folder, cam_name

# ---------- Live preview (จำกัด FPS + ส่งเฉพาะเฟรมใหม่) ----------
def capture_loop():
    global latest_frame, mode, running, supports_preview, selected_port, last_error, preview_fps
    cam = connect_camera(selected_port)
    if cam:
        try_enable_liveview(cam)
        if supports_preview is None:
            supports_preview = _check_preview_support(cam)
        try:
            try_autofocus(cam, timeout_s=0.4)  # โฟกัสสั้น ๆ ตอนเข้า live
        except Exception:
            pass

    # จำกัดอัตราการ "ดึงภาพจากกล้อง"
    frame_interval = 1.0 / max(1.0, float(preview_fps))
    next_tick = time.monotonic() + frame_interval

    while running:
        try:
            if not cam:
                time.sleep(0.4)
                cam = connect_camera(selected_port)
                if cam:
                    try_enable_liveview(cam)
                    if supports_preview is None:
                        supports_preview = _check_preview_support(cam)
                    try:
                        try_autofocus(cam, timeout_s=0.3)
                    except Exception:
                        pass
                next_tick = time.monotonic() + frame_interval
                continue

            if mode == "live":
                if supports_preview is False:
                    try_enable_liveview(cam)
                    supports_preview = _check_preview_support(cam)

                if supports_preview:
                    # หน่วงให้ได้ตาม fps ที่ตั้ง
                    now = time.monotonic()
                    sleep_for = next_tick - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    next_tick += frame_interval
                    if next_tick < time.monotonic() - frame_interval:
                        next_tick = time.monotonic() + frame_interval

                    try:
                        camera_file = cam.capture_preview()
                        file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
                        b = memoryview(file_data).tobytes()
                        # ส่ง JPEG ตรง ๆ ถ้าหัวเป็น SOI (FFD8)
                        if b and b[:2] == b'\xff\xd8':
                            _set_latest_frame(b)
                    except gp.GPhoto2Error as e:
                        print(f"[WARN] live preview failed: {e}")
                        supports_preview = False
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

        except gp.GPhoto2Error as e:
            last_error = f"Camera error, reconnecting: {e}"
            print(f"[WARN] {last_error}")
            try:
                cam.exit()
            except gp.GPhoto2Error:
                pass
            cam = None
            time.sleep(0.8)
            next_tick = time.monotonic() + frame_interval

    try:
        if cam:
            cam.exit()
    except Exception:
        pass
    print("[INFO] capture_loop stopped")

def generate_frames():
    global preview_fps, latest_frame_ver
    send_interval = 1.0 / max(1.0, float(preview_fps))   # จำกัดอัตราส่งให้เบราว์เซอร์
    next_send = time.monotonic()
    last_sent_ver = -1
    boundary = b'--frame\r\n'

    while True:
        # เคารพรอบการส่ง
        now = time.monotonic()
        if now < next_send:
            time.sleep(min(0.005, next_send - now))
            continue

        with lock:
            frame = latest_frame
            ver = latest_frame_ver

        # ส่งเฉพาะ "เฟรมใหม่"
        if frame and ver != last_sent_ver:
            headers = (
                b'Content-Type: image/jpeg\r\n' +
                b'Content-Length: ' + str(len(frame)).encode('ascii') + b'\r\n\r\n'
            )
            yield boundary + headers + frame + b'\r\n'
            last_sent_ver = ver
            next_send = time.monotonic() + send_interval
        else:
            # ยังไม่มีเฟรมใหม่ → หน่วงเบา ๆ
            time.sleep(0.01)

def stop_capture_thread():
    global running, capture_thread
    if capture_thread and capture_thread.is_alive():
        running = False
        capture_thread.join(timeout=2)
    capture_thread = None

def start_capture_thread():
    global running, capture_thread, supports_preview, mode
    if running: return
    supports_preview = None
    mode = "live"
    running = True
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

def has_viewers():
    return viewers > 0

def ensure_preview_if_needed():
    if has_viewers(): start_capture_thread()
    else: stop_capture_thread()

# ---------- Routes ----------
@app.route('/')
def root():
    return "OK", 200

@app.route('/api/health')
def api_health():
    return jsonify({
        "ok": True, "running": running, "viewers": viewers, "mode": mode,
        "selected_port": selected_port, "supports_preview": supports_preview,
        "last_error": last_error, "preview_fps": preview_fps,
    }), 200

@app.route('/api/diag')
def api_diag():
    cams = list_cameras()
    return jsonify({
        "detected": [{"model": m, "port": p} for (m, p) in cams],
        "selected_port": selected_port, "running": running,
        "supports_preview": supports_preview, "viewers": viewers,
        "mode": mode, "last_error": last_error, "preview_fps": preview_fps,
    }), 200

@app.route('/cameras')
def cameras():
    cams = list_cameras()
    return jsonify({
        "cameras": [{"model": m, "port": p} for (m, p) in cams],
        "selected_port": selected_port
    })

@app.route('/set_camera', methods=['GET', 'POST'])
def set_camera():
    global selected_port, supports_preview, mode
    port = request.values.get('camera_port')
    if not port: return "camera_port required", 400
    selected_port = port.strip()
    print(f"[INFO] Switching camera to port {selected_port}")
    supports_preview = None
    mode = "live"
    stop_capture_thread()
    ensure_preview_if_needed()
    return jsonify({"ok": True, "selected_port": selected_port})

@app.route('/capture', methods=['POST'])
def capture():
    global mode, captured_image, captured_filename, selected_port, last_error
    stop_capture_thread()
    cam = connect_camera(selected_port)
    if not cam:
        ensure_preview_if_needed()
        return jsonify({"ok": False, "error": last_error or "Camera not available"}), 503

    # AF ก่อนถ่าย
    try:
        af_ok = try_autofocus(cam, timeout_s=0.6)
        print(f"[INFO] autofocus before capture: {'OK' if af_ok else 'SKIPPED'}")
    except Exception as e:
        print(f"[WARN] autofocus before capture failed: {e}")

    try:
        host_filepath, mime, cam_folder, cam_name = safe_capture_one(cam)
        with open(host_filepath, 'rb') as f:
            captured_image = f.read()
        captured_filename = os.path.abspath(host_filepath)

        pushed = False
        if (mime == 'image/jpeg' or host_filepath.lower().endswith('.jpg')) and captured_image[:2] == b'\xff\xd8':
            _set_latest_frame(captured_image)
            pushed = True
        if not pushed:
            _update_latest_with_camera_preview(cam, cam_folder, cam_name)

        mode = "captured"
        rel_url = f"/captured_images/{os.path.basename(host_filepath)}"
        return jsonify({"ok": True, "url": rel_url, "serverPath": captured_filename}), 200

    except Exception as e:
        last_error = f"Capture failed: {e}"
        print(f"[ERROR] {last_error}")
        return jsonify({"ok": False, "error": last_error}), 500
    finally:
        try: cam.exit()
        except Exception: pass
        with lock:
            if mode == "live": latest_frame = None
        ensure_preview_if_needed()

@app.route('/confirm', methods=['POST'])
def confirm():
    global mode, captured_image, captured_filename, latest_frame
    captured_image = None
    captured_filename = None
    mode = "live"
    stop_capture_thread()
    with lock: latest_frame = None
    ensure_preview_if_needed()
    ts = int(time.time() * 1000)
    return jsonify({"video": f"/video_feed?ts={ts}"}), 200

@app.route('/return_live', methods=['POST'])
def return_live():
    global mode, captured_image, captured_filename, latest_frame
    captured_image = None
    captured_filename = None
    mode = "live"
    stop_capture_thread()
    with lock: latest_frame = None
    ensure_preview_if_needed()
    return "Live", 200

@app.route('/video_feed')
def video_feed():
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
            as_attachment=False,
            download_name=os.path.basename(captured_filename)
        )
        return _nocache_headers(resp)
    else:
        return "No image captured", 404

# --- ให้ตรงกับ stopCamera() ของ UI ---
@app.route('/stop_stream', methods=['POST'])
@app.route('/stop', methods=['POST'])
def stop_stream():
    global mode, latest_frame
    mode = "live"
    stop_capture_thread()
    with lock: latest_frame = None
    return jsonify({"ok": True, "stopped": True}), 200

def cleanup(sig, frame):
    print("\n[INFO] Shutting down server...")
    stop_capture_thread()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == '__main__':
    print(f"[BOOT] FRONTEND_ORIGIN={FRONTEND_ORIGIN}")
    print(f"[BOOT] CAMERA_PORT (env)={CAMERA_PORT_ENV}")
    print(f"[BOOT] Save dir: {SAVE_DIR}")
    print(f"[BOOT] Preview FPS: {preview_fps}")
    print(f"[BOOT] Detected cameras: {list_cameras()}")
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

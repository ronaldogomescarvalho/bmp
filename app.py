"""
rPPG HRV Monitor — Servidor Web
================================
Flask + Flask-SocketIO recebe frames do celular via WebSocket,
processa com MediaPipe + scipy e devolve BPM/HRV em tempo real.
"""

import os
import base64
import time
import numpy as np
from io import BytesIO
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
FILTER_LOW_HZ       = 0.7
FILTER_HIGH_HZ      = 3.5
FILTER_ORDER        = 3
PEAK_MIN_DISTANCE_SEC = 0.3
PEAK_MAX_DISTANCE_SEC = 1.5
MIN_SAMPLES_FOR_BPM = 60     # ~2s a 30fps
MIN_SAMPLES_FOR_HRV = 150    # ~5s a 30fps

FOREHEAD_CENTRAL = [10, 67, 109, 108, 151, 337, 338, 297]

# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "rppg-secret-2024")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─────────────────────────────────────────────
# MEDIAPIPE — carrega o modelo uma vez só
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

base_options = mp_tasks_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_tasks_vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
# ESTADO POR SESSÃO (simples — uma sessão por vez)
# ─────────────────────────────────────────────
sessions = {}


def new_session():
    return {
        "green_signal": [],
        "timestamps": [],
    }


# ─────────────────────────────────────────────
# DSP
# ─────────────────────────────────────────────
def apply_bandpass(signal, fs):
    nyquist = fs / 2.0
    low  = max(FILTER_LOW_HZ  / nyquist, 0.001)
    high = min(FILTER_HIGH_HZ / nyquist, 0.999)
    if low >= high or len(signal) < 15:
        return signal
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    padlen = min(len(signal) - 1, 3 * max(len(b), len(a)))
    return filtfilt(b, a, signal, padlen=padlen)


def estimate_bpm(signal, fs):
    n = len(signal)
    if n < 2:
        return None
    windowed = signal * np.hanning(n)
    yf = np.abs(fft(windowed))[: n // 2]
    xf = fftfreq(n, 1.0 / fs)[: n // 2]
    valid = (xf >= FILTER_LOW_HZ) & (xf <= FILTER_HIGH_HZ)
    if not np.any(valid):
        return None
    peak_freq = xf[valid][np.argmax(yf[valid])]
    return peak_freq * 60.0


def compute_hrv(signal, fs):
    if len(signal) < fs * 5:
        return None, None
    min_dist = max(int(PEAK_MIN_DISTANCE_SEC * fs), 1)
    peaks, _ = find_peaks(
        signal,
        distance=min_dist,
        height=np.median(signal),
        prominence=np.std(signal) * 0.3,
    )
    if len(peaks) < 3:
        return None, None
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr >= PEAK_MIN_DISTANCE_SEC * 1000) & (rr <= PEAK_MAX_DISTANCE_SEC * 1000)]
    if len(rr) < 2:
        return None, None
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    return rr.tolist(), rmssd


def readiness_score(bpm, rmssd):
    if bpm is None or rmssd is None:
        return None
    bpm_score = 50.0 if 55 <= bpm <= 75 else (
        max(0, 50 - (55 - bpm) * 2.5) if bpm < 55 else max(0, 50 - (bpm - 75) * 2.0)
    )
    hrv_score = (
        50.0 if rmssd >= 80 else
        (rmssd - 20) / 60.0 * 50.0 if rmssd >= 20 else
        max(0, rmssd / 20.0 * 10.0)
    )
    return round(min(100, max(0, bpm_score + hrv_score)), 1)


# ─────────────────────────────────────────────
# PROCESSAMENTO DE FRAME
# ─────────────────────────────────────────────
def process_frame(image_data_b64, session):
    """Recebe frame base64, extrai sinal verde, retorna dict com status."""
    try:
        # Decodificar imagem
        header, encoded = image_data_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame = np.array(pil_img)
    except Exception as e:
        return {"error": f"Falha ao decodificar imagem: {e}"}

    h, w, _ = frame.shape

    # MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return {"face_detected": False, "samples": len(session["green_signal"])}

    landmarks = results.face_landmarks[0]

    # ROI da testa
    points = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in FOREHEAD_CENTRAL],
        dtype=np.int32,
    )
    x1, y1 = points[:, 0].min(), points[:, 1].min()
    x2, y2 = points[:, 0].max(), points[:, 1].max()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi_pixels = frame[y1:y2, x1:x2, 1]  # canal verde (RGB)

    if len(roi_pixels) == 0:
        return {"face_detected": True, "samples": len(session["green_signal"])}

    mean_green = float(np.mean(roi_pixels))
    session["green_signal"].append(mean_green)
    session["timestamps"].append(time.time())

    n = len(session["green_signal"])
    result = {"face_detected": True, "samples": n}

    # Calcular FPS real
    if n >= 2:
        elapsed = session["timestamps"][-1] - session["timestamps"][0]
        fs = (n - 1) / elapsed if elapsed > 0 else 30.0
    else:
        fs = 30.0

    result["fps"] = round(fs, 1)

    # BPM em tempo real (a partir de ~2s de dados)
    if n >= MIN_SAMPLES_FOR_BPM:
        sig = np.array(session["green_signal"])
        # detrend + normalizar
        x = np.arange(len(sig))
        sig = sig - np.polyval(np.polyfit(x, sig, 1), x)
        std = np.std(sig)
        if std > 0:
            sig = (sig - np.mean(sig)) / std
            try:
                filtered = apply_bandpass(sig, fs)
                bpm = estimate_bpm(filtered, fs)
                if bpm and 40 <= bpm <= 200:
                    result["bpm"] = round(bpm, 1)
            except Exception:
                pass

    # HRV (a partir de ~5s)
    if n >= MIN_SAMPLES_FOR_HRV:
        try:
            sig = np.array(session["green_signal"])
            x = np.arange(len(sig))
            sig = sig - np.polyval(np.polyfit(x, sig, 1), x)
            std = np.std(sig)
            if std > 0:
                sig = (sig - np.mean(sig)) / std
                filtered = apply_bandpass(sig, fs)
                rr, rmssd = compute_hrv(filtered, fs)
                if rmssd is not None:
                    result["rmssd"] = round(rmssd, 1)
                    result["score"] = readiness_score(result.get("bpm"), rmssd)
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────
# ROTAS HTTP
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    sessions[request_sid()] = new_session()
    emit("connected", {"msg": "Servidor conectado"})


@socketio.on("disconnect")
def on_disconnect():
    sessions.pop(request_sid(), None)


@socketio.on("reset")
def on_reset():
    sessions[request_sid()] = new_session()
    emit("reset_ok")


@socketio.on("frame")
def on_frame(data):
    sid = request_sid()
    if sid not in sessions:
        sessions[sid] = new_session()
    result = process_frame(data["image"], sessions[sid])
    emit("result", result)


def request_sid():
    from flask import request
    return request.sid


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
 socketio.run(app, host="0.0.0.0", port=8080, debug=False, allow_unsafe_werkzeug=True)

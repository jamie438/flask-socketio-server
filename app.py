from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import time
from aubio import pitch
from scipy.signal import butter, lfilter
import logging

logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# 🎚 Bandpass-Filter für bessere Klarinettenanalyse
def bandpass_filter(data, sr, low=100.0, high=1000.0):
    nyq = 0.5 * sr
    low /= nyq
    high /= nyq
    b, a = butter(2, [low, high], btype='band')
    return lfilter(b, a, data)

@app.route("/")
def index():
    return "SocketIO läuft"

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    print(f"📥 Empfangenes audio_chunk-Event um {time.strftime('%H:%M:%S')}", flush=True)

    try:
        if not isinstance(data, dict) or "chunk" not in data:
            print("❌ Ungültige Datenstruktur für 'audio_chunk':", data, flush=True)
            return

        raw_chunk = data["chunk"]
        print(f"🔢 Anzahl Samples: {len(raw_chunk)}", flush=True)

        samples = np.array(raw_chunk, dtype=np.float32)
        if len(samples) == 0:
            print("⚠️ Leerer Audio-Chunk empfangen", flush=True)
            return

        sr_input = 48000
        duration = len(samples) / sr_input
        print(f"⏱️ Dauer: {duration:.3f} Sekunden bei 48kHz", flush=True)

        if duration < 0.1:
            print("⚠️ Zu kurzer Chunk für Pitch-Schätzung", flush=True)
            return

        # Normierung
        peak = np.max(np.abs(samples))
        if peak < 0.01:
            print("⚠️ Zu leises Signal (Peak < 0.01)", flush=True)
            return
        samples = samples / peak

        # 🎚 Filter anwenden
        # 🎚 Filter anwenden
        samples = bandpass_filter(samples, sr_input).astype(np.float32)


        # YIN mit aubio
        win_s = 2048
        hop_s = 512
        p = pitch("yin", win_s, hop_s, sr_input)
        p.set_unit("Hz")
        p.set_silence(-30)

        pitches = []
        for i in range(0, len(samples) - hop_s, hop_s):
            frame = samples[i:i + hop_s]
            f0 = p(frame)[0]
            if 100 < f0 < 2000:  # 🔍 realistische Klarinetten-Töne
                pitches.append(f0)

        if not pitches:
            print("⚠️ Keine gültige Frequenz erkannt", flush=True)
            return

        # Nutze Median oder häufigste Tonhöhe (optional)
        freq = float(np.median(pitches))
        emit("frequency_result", {"frequency": freq})
        print(f"✅ YIN Pitch erkannt & gesendet: {freq:.1f} Hz", flush=True)

    except Exception as e:
        print("❌ Fehler bei der Analyse:", str(e), flush=True)

if __name__ == "__main__":
    print("🚀 Starte Flask-Server auf Port 8000", flush=True)
    socketio.run(app, host="0.0.0.0", port=8000)

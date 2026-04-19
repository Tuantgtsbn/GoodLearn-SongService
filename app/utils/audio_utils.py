"""
Audio Utilities
- Tạo audio test (sine wave, chirp)
- Convert định dạng
- Thông tin audio file
"""
import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple


def generate_test_audio_sine(
    frequency: float = 440.0,
    duration: float = 5.0,
    sr: int = 22050,
    output_path: str = "test_sine.wav",
) -> str:
    """
    Tạo audio sine wave test (nốt A4 mặc định).
    Dùng để kiểm tra pipeline chấm điểm.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Thêm chút vibrato nhẹ để giống giọng người hơn
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5.5 * t)
    y = 0.5 * np.sin(2 * np.pi * frequency * vibrato * t)

    # Thêm envelope ADSR đơn giản
    attack = int(0.1 * sr)
    release = int(0.3 * sr)
    envelope = np.ones(len(y))
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    y = (y * envelope).astype(np.float32)

    sf.write(output_path, y, sr)
    return output_path


def generate_test_audio_melody(
    sr: int = 22050,
    output_path: str = "test_melody.wav",
) -> str:
    """
    Tạo giai điệu đơn giản (Do Re Mi) để test.
    """
    # Tần số các nốt: C4, D4, E4, F4, G4, A4, B4, C5
    notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    note_duration = 0.5  # giây mỗi nốt

    segments = []
    for freq in notes:
        t = np.linspace(0, note_duration, int(sr * note_duration), endpoint=False)
        # Vibrato nhẹ
        vibrato = 1 + 0.015 * np.sin(2 * np.pi * 5.8 * t)
        seg = 0.4 * np.sin(2 * np.pi * freq * vibrato * t)

        # Envelope
        attack = int(0.05 * sr)
        release = int(0.1 * sr)
        env = np.ones(len(seg))
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        segments.append((seg * env).astype(np.float32))

    y = np.concatenate(segments)
    sf.write(output_path, y, sr)
    return output_path


def get_audio_info(file_path: str) -> dict:
    """Lấy thông tin cơ bản của file audio"""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        duration = len(y) / sr
        return {
            "file_path": file_path,
            "file_size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 3),
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "num_samples": len(y),
            "channels": 1,  # đã mono
        }
    except Exception as e:
        return {"error": str(e)}


def hz_to_note_name(hz: float) -> str:
    """Chuyển tần số Hz → tên nốt (vd: 440 → A4)"""
    if hz <= 0:
        return "N/A"
    try:
        midi = librosa.hz_to_midi(hz)
        return librosa.midi_to_note(int(round(float(midi))))
    except Exception:
        return "N/A"

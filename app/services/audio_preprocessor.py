"""
Audio Preprocessing Service
- Load và chuẩn hóa audio
- Phát hiện giọng nói (Voice Activity Detection)
- Denoising cơ bản
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import os

from app.core.config import settings
from app.core.logger import logger


class AudioPreprocessor:
    """Xử lý và chuẩn hóa audio đầu vào"""

    def __init__(self):
        self.sr = settings.SAMPLE_RATE
        self.hop_length = settings.HOP_LENGTH
        self.n_fft = settings.N_FFT

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file và resample về sample_rate chuẩn.
        Hỗ trợ: .wav, .mp3, .ogg, .flac, .m4a
        """
        logger.info(f"Đang load audio: {file_path}")
        try:
            y, sr = librosa.load(
                file_path,
                sr=self.sr,
                mono=True,  # Chuyển về mono
                dtype=np.float32,
            )
        except Exception as e:
            raise ValueError(f"Không thể đọc file audio: {e}")

        duration = len(y) / sr
        if duration > settings.MAX_AUDIO_DURATION:
            raise ValueError(
                f"Audio quá dài ({duration:.1f}s). Tối đa {settings.MAX_AUDIO_DURATION}s."
            )
        if duration < 1.0:
            raise ValueError("Audio quá ngắn (tối thiểu 1 giây).")

        logger.info(f"Audio loaded: {duration:.2f}s, sr={sr}Hz, samples={len(y)}")
        return y, sr

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Chuẩn hóa biên độ về [-1, 1]"""
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.9  # 0.9 để tránh clipping
        return y

    def remove_silence(
        self, y: np.ndarray, top_db: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loại bỏ phần im lặng đầu/cuối.
        Trả về: (y_trimmed, intervals) - audio đã trim và khoảng có âm thanh
        """
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed

    def voice_activity_detection(
        self, y: np.ndarray, frame_length: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """
        Phát hiện các frame có giọng nói.
        Trả về mask boolean: True = có giọng, False = im lặng
        """
        # Tính năng lượng RMS từng frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        # Ngưỡng: 20% giá trị RMS lớn nhất
        threshold = np.percentile(rms, 20)
        voiced_mask = rms > threshold
        return voiced_mask, rms

    def compute_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Tính mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128,
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def preprocess(self, file_path: str) -> dict:
        """
        Pipeline xử lý hoàn chỉnh.
        Trả về dict chứa audio data và metadata.
        """
        y, sr = self.load_audio(file_path)
        y = self.normalize_audio(y)
        y_trimmed = self.remove_silence(y)
        voiced_mask, rms = self.voice_activity_detection(y_trimmed)

        voiced_ratio = float(np.sum(voiced_mask)) / len(voiced_mask) if len(voiced_mask) > 0 else 0.0

        return {
            "y": y_trimmed,
            "y_raw": y,
            "sr": sr,
            "duration": len(y_trimmed) / sr,
            "raw_duration": len(y) / sr,
            "voiced_mask": voiced_mask,
            "voiced_ratio": voiced_ratio,
            "rms": rms,
        }


# Singleton instance
preprocessor = AudioPreprocessor()

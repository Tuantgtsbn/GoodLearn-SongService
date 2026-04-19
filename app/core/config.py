"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Vocal Scoring API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = "sqlite:///./vocal_scoring.db"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_SONGS: str = "songs"

    # Audio Processing
    SAMPLE_RATE: int = 22050           # Hz - tần số lấy mẫu chuẩn
    HOP_LENGTH: int = 512              # Bước nhảy STFT
    N_FFT: int = 2048                  # Kích thước FFT window
    MAX_AUDIO_DURATION: float = 400.0  # Giây - tối đa 6 phút 40 giây
    MAX_FILE_SIZE_MB: int = 50         # MB

    # Scoring Weights (tổng = 1.0)
    WEIGHT_PITCH: float = 0.34         # Độ chuẩn cao độ
    WEIGHT_RHYTHM: float = 0.24        # Độ chuẩn nhịp điệu
    WEIGHT_STABILITY: float = 0.12     # Sự ổn định giọng
    WEIGHT_DYNAMICS: float = 0.30      # Diễn đạt cảm xúc (dynamics)

    # Pitch Detection
    PITCH_FMIN: float = 80.0           # Hz - tần số thấp nhất (giọng nam thấp)
    PITCH_FMAX: float = 1200.0         # Hz - tần số cao nhất (giọng nữ cao)
    PITCH_CONFIDENCE_THRESHOLD: float = 0.5  # Ngưỡng tin cậy pitch
    PITCH_JUMP_THRESHOLD: float = 0.8
    PITCH_JUMP_PENALTY_SCALE: float = 6.0
    PITCH_JUMP_RATIO_SCALE: float = 70.0
    PITCH_JITTER_PENALTY_SCALE: float = 1.6
    PITCH_JITTER_PENALTY_CAP: float = 20.0

    # Stability penalty scales
    STABILITY_JITTER_SCALE: float = 14.0
    STABILITY_SHIMMER_SCALE: float = 7.0
    STABILITY_BREATHINESS_SCALE: float = 45.0
    STABILITY_TREMOLO_PENALTY: float = 12.0

    # Feature Flags
    ENABLE_VOCAL_SEPARATION: bool = True
    ENABLE_DTW_ALIGNMENT: bool = True

    # Vocal Separation
    VOCAL_SEPARATOR_BACKEND: str = "auto"    # auto|demucs|spleeter|none
    VOCAL_SEPARATOR_DEVICE: str = "auto"     # auto|cpu|cuda
    VOCAL_SEPARATOR_MAX_DURATION_SECONDS: float = 90.0
    DEMUCS_TIMEOUT_SECONDS: int = 25
    SKIP_REFERENCE_SEPARATION: bool = True

    # DTW Alignment
    DTW_MAX_FRAMES: int = 3000
    DTW_PENALTY: float = 0.1

    # Directories
    UPLOAD_DIR: str = "uploads"
    REFERENCE_SONGS_DIR: str = "data/reference_songs"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Tạo thư mục nếu chưa có
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REFERENCE_SONGS_DIR, exist_ok=True)

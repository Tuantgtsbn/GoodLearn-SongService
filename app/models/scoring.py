"""
Pydantic Models - Request & Response Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ScoreGrade(str, Enum):
    S = "S"       # 90-100: Xuất sắc
    A = "A"       # 80-89:  Giỏi
    B = "B"       # 70-79:  Khá
    C = "C"       # 60-69:  Trung bình
    D = "D"       # 50-59:  Yếu
    F = "F"       # 0-49:   Kém


class VocalRange(str, Enum):
    BASS = "Bass"
    BARITONE = "Baritone"
    TENOR = "Tenor"
    ALTO = "Alto"
    MEZZO_SOPRANO = "Mezzo-Soprano"
    SOPRANO = "Soprano"
    UNKNOWN = "Unknown"


# ─────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────

class PitchAnalysis(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Điểm cao độ (0-100)")
    average_pitch_hz: float = Field(..., description="Cao độ trung bình (Hz)")
    average_pitch_note: str = Field(..., description="Nốt nhạc trung bình")
    pitch_accuracy_percent: float = Field(..., description="% nốt nhạc đúng cao độ")
    out_of_tune_segments: int = Field(..., description="Số đoạn lạc giọng")
    pitch_stability: float = Field(..., description="Độ ổn định cao độ (0-1)")
    vocal_range: VocalRange = Field(..., description="Tầm âm giọng hát")
    voiced_ratio: float = Field(..., description="Tỷ lệ có âm thanh/tổng thời gian")


class RhythmAnalysis(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Điểm nhịp điệu (0-100)")
    estimated_tempo_bpm: float = Field(..., description="Tempo ước tính (BPM)")
    beat_consistency: float = Field(..., description="Độ nhất quán của beat (0-1)")
    onset_regularity: float = Field(..., description="Độ đều đặn onset (0-1)")
    rhythm_deviation_ms: float = Field(..., description="Sai lệch nhịp trung bình (ms)")


class StabilityAnalysis(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Điểm ổn định (0-100)")
    vibrato_rate_hz: Optional[float] = Field(None, description="Tốc độ rung (Hz)")
    vibrato_extent_semitones: Optional[float] = Field(None, description="Biên độ rung (semitones)")
    tremolo_detected: bool = Field(..., description="Phát hiện run giọng bất thường")
    breathiness_score: float = Field(..., description="Mức độ hơi thở (0-1, thấp hơn = tốt hơn)")
    jitter_percent: float = Field(..., description="Jitter - biến động chu kỳ (%)")
    shimmer_percent: float = Field(..., description="Shimmer - biến động biên độ (%)")


class DynamicsAnalysis(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Điểm diễn đạt (0-100)")
    dynamic_range_db: float = Field(..., description="Dải động lực (dB)")
    loudness_variation: float = Field(..., description="Biến động âm lượng")
    emotional_expressiveness: float = Field(..., description="Mức độ biểu cảm (0-1)")
    rms_energy_mean: float = Field(..., description="Năng lượng RMS trung bình")


class SegmentScore(BaseModel):
    start_time: float = Field(..., description="Thời điểm bắt đầu (giây)")
    end_time: float = Field(..., description="Thời điểm kết thúc (giây)")
    pitch_score: float = Field(..., description="Điểm pitch đoạn này")
    overall_score: float = Field(..., description="Điểm tổng đoạn này")
    feedback: str = Field(..., description="Nhận xét cho đoạn này")


class Feedback(BaseModel):
    strengths: List[str] = Field(..., description="Điểm mạnh")
    improvements: List[str] = Field(..., description="Cần cải thiện")
    tips: List[str] = Field(..., description="Lời khuyên luyện tập")
    overall_comment: str = Field(..., description="Nhận xét tổng thể")


# ─────────────────────────────────────────────
# Main Response
# ─────────────────────────────────────────────

class ScoringResult(BaseModel):
    # Metadata
    song_title: Optional[str] = Field(None, description="Tên bài hát")
    audio_duration_seconds: float = Field(..., description="Thời lượng audio (giây)")
    sample_rate: int = Field(..., description="Tần số lấy mẫu")

    # Scores
    total_score: float = Field(..., ge=0, le=100, description="Tổng điểm (0-100)")
    grade: ScoreGrade = Field(..., description="Xếp loại")

    # Detailed Analysis
    pitch: PitchAnalysis
    rhythm: RhythmAnalysis
    stability: StabilityAnalysis
    dynamics: DynamicsAnalysis

    # Segment breakdown
    segments: List[SegmentScore] = Field(default_factory=list)

    # Human feedback
    feedback: Feedback

    # Processing info
    processing_time_ms: float = Field(..., description="Thời gian xử lý (ms)")


class ScoringError(BaseModel):
    error: str
    detail: str
    code: str

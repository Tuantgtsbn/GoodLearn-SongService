"""
Stability & Dynamics Analysis Engine
- Vocal stability (vibrato, tremolo, breathiness)
- Dynamics / expressiveness analysis
"""
import numpy as np
import librosa
from typing import Tuple, Optional

from app.core.config import settings
from app.core.logger import logger
from app.models.scoring import StabilityAnalysis, DynamicsAnalysis


class StabilityAnalyzer:
    """Phân tích sự ổn định của giọng hát"""

    def __init__(self):
        self.sr = settings.SAMPLE_RATE
        self.hop_length = settings.HOP_LENGTH

    def detect_tremolo(self, f0: np.ndarray, voiced_flag: np.ndarray) -> bool:
        """
        Phát hiện tremolo (rung không đều, không có chủ đích).
        Tremolo: dao động > 8 Hz với biên độ > 1 semitone.
        """
        voiced_f0 = f0[voiced_flag & (f0 > 0)]
        if len(voiced_f0) < 30:
            return False

        midi_vals = 69 + 12 * np.log2(np.maximum(voiced_f0, 1e-6) / 440)
        detrended = midi_vals - np.convolve(midi_vals, np.ones(5) / 5, mode="same")

        fft_vals = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended), d=self.hop_length / self.sr)

        tremolo_range = (freqs > 8) & (freqs < 15)
        if not np.any(tremolo_range):
            return False

        tremolo_energy = np.sum(fft_vals[tremolo_range])
        total_energy = np.sum(fft_vals) + 1e-6
        return bool(tremolo_energy / total_energy > 0.3)

    def compute_breathiness(self, y: np.ndarray, voiced_flag: np.ndarray) -> float:
        """
        Đo mức độ hơi thở trong giọng hát.
        Dùng tỷ lệ năng lượng tần số cao / tổng (HNR proxy).
        Breathiness cao → giọng hơi, khàn.
        """
        # Spectral flatness: cao = nhiều noise (hơi thở), thấp = tonal
        spec_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        voiced_flatness = spec_flatness[:len(voiced_flag)][voiced_flag[:len(spec_flatness)]]

        if len(voiced_flatness) == 0:
            return 0.3

        # Chuyển về thang 0-1 (0 = không có hơi, 1 = rất nhiều hơi)
        mean_flatness = float(np.mean(voiced_flatness))
        breathiness = float(np.clip(mean_flatness * 10, 0, 1))
        return breathiness

    def compute_spectral_stability(self, y: np.ndarray) -> float:
        """Đo tính ổn định quang phổ âm thanh"""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel_spec)

        # Frame-to-frame variation
        frame_diffs = np.diff(mel_db, axis=1)
        variation = float(np.mean(np.abs(frame_diffs)))

        # Chuẩn hóa: thấp = ổn định
        stability = float(np.clip(1.0 - variation / 20.0, 0, 1))
        return stability

    def score_stability(
        self,
        jitter: float,
        shimmer: float,
        breathiness: float,
        tremolo_detected: bool,
        vibrato_rate: Optional[float],
        vibrato_extent: Optional[float],
    ) -> float:
        """Tính điểm ổn định"""
        # Jitter penalty: > 1% là đáng lo, > 3% là kém
        jitter_score = float(np.clip(100 - jitter * settings.STABILITY_JITTER_SCALE, 0, 100))

        # Shimmer penalty: > 3% là đáng lo
        shimmer_score = float(np.clip(100 - shimmer * settings.STABILITY_SHIMMER_SCALE, 0, 100))

        # Breathiness penalty
        breath_score = float(np.clip(100 - breathiness * settings.STABILITY_BREATHINESS_SCALE, 0, 100))

        # Tremolo penalty
        tremolo_penalty = settings.STABILITY_TREMOLO_PENALTY if tremolo_detected else 0.0

        # Vibrato bonus/penalty
        vibrato_bonus = 0.0
        if vibrato_rate is not None and vibrato_extent is not None:
            if 4.5 <= vibrato_rate <= 7.5 and 0.3 <= vibrato_extent <= 1.5:
                vibrato_bonus = 8.0  # Vibrato đẹp
            elif vibrato_extent > 2.0:
                vibrato_bonus = -10.0  # Vibrato quá rộng

        base_score = (jitter_score * 0.35 + shimmer_score * 0.35 + breath_score * 0.30)
        total = base_score - tremolo_penalty + vibrato_bonus
        return float(np.clip(total, 0, 100))

    def analyze(
        self,
        y: np.ndarray,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        jitter: float,
        shimmer: float,
        vibrato_rate: Optional[float],
        vibrato_extent: Optional[float],
    ) -> StabilityAnalysis:
        """Phân tích độ ổn định hoàn chỉnh"""
        logger.info("Đang phân tích độ ổn định...")

        tremolo = self.detect_tremolo(f0, voiced_flag)
        breathiness = self.compute_breathiness(y, voiced_flag)
        score = self.score_stability(
            jitter, shimmer, breathiness, tremolo, vibrato_rate, vibrato_extent
        )

        logger.info(
            f"Stability score: {score:.1f}, jitter={jitter:.2f}%, "
            f"shimmer={shimmer:.2f}%, breathiness={breathiness:.3f}, tremolo={tremolo}"
        )

        return StabilityAnalysis(
            score=round(score, 2),
            vibrato_rate_hz=round(vibrato_rate, 2) if vibrato_rate else None,
            vibrato_extent_semitones=round(vibrato_extent, 2) if vibrato_extent else None,
            tremolo_detected=tremolo,
            breathiness_score=round(breathiness, 4),
            jitter_percent=round(jitter, 4),
            shimmer_percent=round(shimmer, 4),
        )


class DynamicsAnalyzer:
    """Phân tích diễn đạt cảm xúc và dynamics"""

    def __init__(self):
        self.sr = settings.SAMPLE_RATE
        self.hop_length = settings.HOP_LENGTH

    def compute_dynamic_range(self, y: np.ndarray) -> float:
        """Tính dải động lực (dB) - khác biệt giữa to nhất và nhỏ nhất"""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_positive = rms[rms > 1e-6]
        if len(rms_positive) < 2:
            return 0.0
        rms_db = librosa.amplitude_to_db(rms_positive)
        p5  = float(np.percentile(rms_db, 5))
        p95 = float(np.percentile(rms_db, 95))
        return float(p95 - p5)

    def compute_loudness_variation(self, y: np.ndarray) -> float:
        """Đo biến động âm lượng theo thời gian (0-1)"""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        if len(rms) < 2:
            return 0.0
        variation = float(np.std(rms) / (np.mean(rms) + 1e-6))
        return float(np.clip(variation, 0, 2.0))

    def compute_expressiveness(self, y: np.ndarray) -> float:
        """
        Đo tính biểu cảm dựa trên:
        - Spectral centroid variation (thay đổi âm sắc)
        - RMS variation (thay đổi âm lượng)
        """
        # Spectral centroid (âm sắc)
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)[0]
        centroid_var = float(np.std(centroid) / (np.mean(centroid) + 1e-6))

        # RMS variation
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_var = float(np.std(rms) / (np.mean(rms) + 1e-6))

        # Zero crossing rate variation
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
        zcr_var = float(np.std(zcr) / (np.mean(zcr) + 1e-6))

        expressiveness = np.clip(
            (centroid_var * 0.4 + rms_var * 0.4 + zcr_var * 0.2) / 0.5,
            0, 1
        )
        return float(expressiveness)

    def score_dynamics(
        self,
        dynamic_range_db: float,
        loudness_variation: float,
        expressiveness: float,
    ) -> float:
        """
        Tính điểm dynamics.
        - Dynamic range lý tưởng: 15-35 dB
        - Variation: vừa phải (không quá phẳng, không quá loạn)
        - Expressiveness: càng cao càng tốt
        """
        # Dynamic range score
        if dynamic_range_db < 5:
            dr_score = 30.0   # Quá phẳng (robot)
        elif dynamic_range_db < 15:
            dr_score = 60.0
        elif dynamic_range_db <= 35:
            dr_score = 90.0   # Lý tưởng
        elif dynamic_range_db <= 50:
            dr_score = 75.0
        else:
            dr_score = 50.0   # Quá biến động

        # Loudness variation score
        if loudness_variation < 0.1:
            lv_score = 40.0   # Quá đều/monotone
        elif loudness_variation < 0.5:
            lv_score = 85.0   # Tốt
        elif loudness_variation < 1.0:
            lv_score = 70.0   # Hơi loạn
        else:
            lv_score = 50.0   # Quá loạn

        # Expressiveness score
        exp_score = float(expressiveness * 100)

        total = dr_score * 0.35 + lv_score * 0.30 + exp_score * 0.35
        return float(np.clip(total, 0, 100))

    def analyze(self, y: np.ndarray) -> DynamicsAnalysis:
        """Phân tích dynamics hoàn chỉnh"""
        logger.info("Đang phân tích dynamics...")

        dynamic_range = self.compute_dynamic_range(y)
        loudness_variation = self.compute_loudness_variation(y)
        expressiveness = self.compute_expressiveness(y)
        rms_mean = float(np.mean(librosa.feature.rms(y=y, hop_length=self.hop_length)[0]))
        score = self.score_dynamics(dynamic_range, loudness_variation, expressiveness)

        logger.info(
            f"Dynamics score: {score:.1f}, range={dynamic_range:.1f}dB, "
            f"expressiveness={expressiveness:.3f}"
        )

        return DynamicsAnalysis(
            score=round(score, 2),
            dynamic_range_db=round(dynamic_range, 2),
            loudness_variation=round(loudness_variation, 4),
            emotional_expressiveness=round(expressiveness, 4),
            rms_energy_mean=round(float(rms_mean), 6),
        )


stability_analyzer = StabilityAnalyzer()
dynamics_analyzer = DynamicsAnalyzer()

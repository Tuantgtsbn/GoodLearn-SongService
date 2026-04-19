"""
Rhythm Analysis Engine
- Tempo estimation (BPM)
- Beat tracking
- Onset detection
- Rhythm consistency scoring
"""
import numpy as np
import librosa
from typing import Tuple, Optional

from app.core.config import settings
from app.core.logger import logger
from app.models.scoring import RhythmAnalysis


class RhythmAnalyzer:
    """Phân tích nhịp điệu và độ chuẩn nhịp"""

    def __init__(self):
        self.sr = settings.SAMPLE_RATE
        self.hop_length = settings.HOP_LENGTH

    def estimate_tempo(self, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Ước tính tempo (BPM) và vị trí các beat.
        Returns: (tempo_bpm, beat_frames)
        """
        # Tính onset strength envelope
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
        )

        # Beat tracking với dynamic programming
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length,
            trim=False,
        )

        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        logger.debug(f"Tempo ước tính: {tempo:.1f} BPM, {len(beats)} beats")
        return tempo, beats, onset_env

    def compute_beat_consistency(self, beats: np.ndarray) -> float:
        """
        Đo độ nhất quán của beat.
        Beat đều đặn → consistency cao → điểm cao.
        """
        if len(beats) < 4:
            return 0.5

        # Khoảng cách giữa các beat (frames)
        intervals = np.diff(beats)
        if len(intervals) == 0:
            return 0.5

        mean_interval = np.mean(intervals)
        std_interval  = np.std(intervals)

        # Coefficient of variation: thấp = đều đặn
        cv = std_interval / mean_interval if mean_interval > 0 else 1.0
        consistency = float(np.clip(1.0 - cv, 0, 1))
        return consistency

    def detect_onsets(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phát hiện onset (điểm bắt đầu âm thanh).
        Returns: (onset_times, onset_strengths)
        """
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=True,
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sr, hop_length=self.hop_length
        )

        # Strength tại mỗi onset
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])

        return onset_times, onset_strengths

    def compute_onset_regularity(self, onset_times: np.ndarray) -> float:
        """
        Đo độ đều đặn của onset (liên quan đến nhịp và phách).
        """
        if len(onset_times) < 4:
            return 0.5

        intervals = np.diff(onset_times)
        if len(intervals) == 0:
            return 0.5

        mean_interval = np.mean(intervals)
        std_interval  = np.std(intervals)

        # Loại bỏ outlier (ngắt hơi quá dài)
        q75, q25 = np.percentile(intervals, [75, 25])
        iqr = q75 - q25
        filtered = intervals[(intervals >= q25 - 1.5 * iqr) & (intervals <= q75 + 1.5 * iqr)]

        if len(filtered) < 2:
            return 0.5

        filtered_std = np.std(filtered)
        filtered_mean = np.mean(filtered)
        cv = filtered_std / filtered_mean if filtered_mean > 0 else 1.0
        return float(np.clip(1.0 - cv * 2, 0, 1))

    def compute_rhythm_deviation(
        self, onset_times: np.ndarray, tempo_bpm: float
    ) -> float:
        """
        Tính sai lệch trung bình so với nhịp lý tưởng (ms).
        """
        if tempo_bpm <= 0 or len(onset_times) < 2:
            return 100.0

        beat_period = 60.0 / tempo_bpm  # giây/beat

        # Tìm khoảng cách từ mỗi onset đến beat gần nhất
        deviations = []
        for t in onset_times:
            nearest_beat_phase = t % beat_period
            deviation = min(nearest_beat_phase, beat_period - nearest_beat_phase)
            deviations.append(deviation)

        mean_dev_ms = float(np.mean(deviations)) * 1000  # → milliseconds
        return round(mean_dev_ms, 2)

    @staticmethod
    def _normalize_onset_times(onset_times: np.ndarray) -> np.ndarray:
        if len(onset_times) == 0:
            return onset_times
        return onset_times - onset_times[0]

    def compute_reference_rhythm_deviation(
        self,
        user_onset_times: np.ndarray,
        ref_onset_times: np.ndarray,
    ) -> float:
        """
        Tính sai lệch nhịp user so với reference.
        Có bù lệch điểm vào bài (start offset) bằng cách normalize mốc thời gian đầu.
        """
        if len(user_onset_times) < 2 or len(ref_onset_times) < 2:
            return 100.0

        user_norm = self._normalize_onset_times(user_onset_times)
        ref_norm = self._normalize_onset_times(ref_onset_times)

        if settings.ENABLE_DTW_ALIGNMENT:
            try:
                cost_matrix = np.abs(user_norm[:, None] - ref_norm[None, :]).astype(np.float32)
                _, wp = librosa.sequence.dtw(C=cost_matrix)
                wp = np.asarray(wp)[::-1]

                aligned_user = user_norm[wp[:, 0]]
                aligned_ref = ref_norm[wp[:, 1]]
                deviation_ms = float(np.mean(np.abs(aligned_user - aligned_ref)) * 1000.0)
                return round(deviation_ms, 2)
            except Exception as exc:
                logger.warning(f"DTW rhythm alignment failed, fallback naive: {exc}")

        # Fallback: index-based alignment after start-offset normalization.
        min_len = min(len(user_norm), len(ref_norm))
        if min_len < 2:
            return 100.0

        naive_ms = float(np.mean(np.abs(user_norm[:min_len] - ref_norm[:min_len])) * 1000.0)
        return round(naive_ms, 2)

    def score_rhythm(
        self,
        beat_consistency: float,
        onset_regularity: float,
        rhythm_deviation_ms: float,
        tempo_bpm: float,
    ) -> float:
        """
        Tính điểm nhịp tổng hợp.
        """
        # Điểm beat consistency (0-100)
        beat_score = beat_consistency * 100

        # Điểm onset regularity (0-100)
        onset_score = onset_regularity * 100

        # Điểm deviation (sai lệch ít = điểm cao)
        # deviation < 30ms: tốt, < 80ms: chấp nhận, > 150ms: kém
        dev_score = float(np.clip(100 - rhythm_deviation_ms * 0.5, 0, 100))

        # Bonus nếu tempo trong khoảng nhạc thường (60-180 BPM)
        tempo_bonus = 5.0 if 60 <= tempo_bpm <= 180 else 0.0

        total = beat_score * 0.4 + onset_score * 0.35 + dev_score * 0.25 + tempo_bonus
        return float(np.clip(total, 0, 100))

    def analyze(self, y: np.ndarray, reference_y: Optional[np.ndarray] = None) -> RhythmAnalysis:
        """Phân tích nhịp điệu hoàn chỉnh"""
        logger.info("Đang phân tích nhịp điệu...")

        tempo, beats, onset_env = self.estimate_tempo(y)
        beat_consistency = self.compute_beat_consistency(beats)
        onset_times, onset_strengths = self.detect_onsets(y)
        onset_regularity = self.compute_onset_regularity(onset_times)

        if reference_y is not None:
            ref_onset_times, _ = self.detect_onsets(reference_y)
            rhythm_deviation_ms = self.compute_reference_rhythm_deviation(onset_times, ref_onset_times)
        else:
            rhythm_deviation_ms = self.compute_rhythm_deviation(onset_times, tempo)

        score = self.score_rhythm(beat_consistency, onset_regularity, rhythm_deviation_ms, tempo)

        logger.info(
            f"Rhythm score: {score:.1f}, tempo={tempo:.1f} BPM, "
            f"consistency={beat_consistency:.3f}, deviation={rhythm_deviation_ms:.1f}ms"
        )

        return RhythmAnalysis(
            score=round(score, 2),
            estimated_tempo_bpm=round(tempo, 2),
            beat_consistency=round(beat_consistency, 4),
            onset_regularity=round(onset_regularity, 4),
            rhythm_deviation_ms=round(rhythm_deviation_ms, 2),
        )


rhythm_analyzer = RhythmAnalyzer()

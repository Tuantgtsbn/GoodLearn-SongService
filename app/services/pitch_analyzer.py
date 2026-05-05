"""
Pitch Analysis Engine
- Phát hiện cao độ (F0) bằng pyin algorithm
- Phân tích độ chuẩn cao độ
- Xác định tầm âm giọng hát
- Chuyển Hz → tên nốt nhạc
"""
import numpy as np
import librosa
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings("ignore")

from app.core.config import settings
from app.core.logger import logger
from app.models.scoring import PitchAnalysis, VocalRange
from app.services.time_aligner import time_aligner


# Ánh xạ tần số → tầm âm giọng hát
VOCAL_RANGE_BOUNDARIES = {
    VocalRange.BASS:          (80,  320),
    VocalRange.BARITONE:      (98,  392),
    VocalRange.TENOR:         (130, 523),
    VocalRange.ALTO:          (175, 698),
    VocalRange.MEZZO_SOPRANO: (196, 880),
    VocalRange.SOPRANO:       (246, 1175),
}


class PitchAnalyzer:
    """Phân tích cao độ và độ chuẩn giọng hát"""

    def __init__(self):
        self.sr = settings.SAMPLE_RATE
        self.hop_length = settings.HOP_LENGTH
        self.fmin = settings.PITCH_FMIN
        self.fmax = settings.PITCH_FMAX
        self.confidence_threshold = settings.PITCH_CONFIDENCE_THRESHOLD

    # ─────────────────────────────────────────
    # Pitch Extraction
    # ─────────────────────────────────────────

    def extract_pitch(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất F0 (fundamental frequency) bằng pyin.
        
        Returns:
            f0: mảng tần số Hz (0 = unvoiced)
            voiced_flag: boolean mask
            voiced_probs: xác suất voiced [0, 1]
        """
        duration = len(y) / self.sr
        logger.debug(f"Đang trích xuất F0 bằng pyin... (duration={duration:.1f}s, samples={len(y)})")
        try:
            # Xử lý theo từng chunk 10s để tránh OOM làm chết worker
            CHUNK_SECONDS = 10
            chunk_length = CHUNK_SECONDS * self.sr  # samples per chunk
            
            if len(y) <= chunk_length:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    fill_na=0.0,
                )
            else:
                import gc
                f0_list, voiced_flag_list, voiced_probs_list = [], [], []
                total_chunks = (len(y) + chunk_length - 1) // chunk_length
                
                for idx, i in enumerate(range(0, len(y), chunk_length)):
                    y_chunk = y[i:i + chunk_length]
                    logger.debug(f"  pyin chunk {idx+1}/{total_chunks} ({len(y_chunk)/self.sr:.1f}s)")
                    
                    f0_c, voiced_flag_c, voiced_probs_c = librosa.pyin(
                        y_chunk,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        sr=self.sr,
                        hop_length=self.hop_length,
                        fill_na=0.0,
                    )
                    
                    f0_list.append(f0_c)
                    voiced_flag_list.append(voiced_flag_c)
                    voiced_probs_list.append(voiced_probs_c)
                    
                    # Giải phóng bộ nhớ sau mỗi chunk
                    del y_chunk
                    gc.collect()
                
                f0 = np.concatenate(f0_list)
                voiced_flag = np.concatenate(voiced_flag_list)
                voiced_probs = np.concatenate(voiced_probs_list)
                
                # Giải phóng lists
                del f0_list, voiced_flag_list, voiced_probs_list
                gc.collect()

            logger.debug(f"Trích xuất F0 xong: {len(f0)} frames")

        except Exception as e:
            logger.warning(f"pyin thất bại, fallback về piptrack: {e}")
            f0, voiced_flag, voiced_probs = self._piptrack_fallback(y)

        return f0, voiced_flag, voiced_probs

    def _piptrack_fallback(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback: dùng piptrack khi pyin không khả dụng"""
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        f0 = np.zeros(pitches.shape[1])
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            if magnitudes[index, t] > 0:
                f0[t] = pitches[index, t]

        voiced_flag = f0 > 0
        voiced_probs = (f0 > 0).astype(float)
        return f0, voiced_flag, voiced_probs

    # ─────────────────────────────────────────
    # Note Conversion
    # ─────────────────────────────────────────

    @staticmethod
    def hz_to_note(hz: float) -> str:
        """Chuyển Hz → tên nốt nhạc (ví dụ: 440 → A4)"""
        if hz <= 0:
            return "N/A"
        try:
            midi = librosa.hz_to_midi(hz)
            note = librosa.midi_to_note(int(round(midi)))
            return note
        except Exception:
            return "N/A"

    @staticmethod
    def hz_to_midi(hz: float) -> float:
        """Chuyển Hz → MIDI number (cho phép số thực)"""
        if hz <= 0:
            return 0.0
        return 69.0 + 12.0 * np.log2(hz / 440.0)

    @staticmethod
    def cents_difference(f1: float, f2: float) -> float:
        """Tính sai lệch giữa 2 tần số theo cents (100 cents = 1 semitone)"""
        if f1 <= 0 or f2 <= 0:
            return 0.0
        return 1200.0 * np.log2(f1 / f2)

    # ─────────────────────────────────────────
    # Pitch Accuracy Scoring
    # ─────────────────────────────────────────

    def score_pitch_accuracy(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        reference_f0: Optional[np.ndarray] = None,
    ) -> float:
        """
        Chấm điểm độ chuẩn cao độ.
        
        Nếu có reference_f0: so sánh với bài hát gốc.
        Nếu không: đánh giá dựa trên tính nhất quán nội bộ.
        """
        voiced_f0 = f0[voiced_flag & (f0 > 0)]
        if len(voiced_f0) < 5:
            return 50.0  # Không đủ dữ liệu

        if reference_f0 is not None:
            return self._score_against_reference(f0, voiced_flag, reference_f0)
        else:
            return self._score_internal_consistency(voiced_f0)

    def _score_against_reference(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        ref_f0: np.ndarray,
    ) -> float:
        """So sánh với bài hát gốc (DTW nếu bật, fallback naive nếu cần)."""
        aligned = time_aligner.align_pitch_tracks(f0, voiced_flag, ref_f0)
        if aligned is not None:
            cents_diff = np.abs([
                self.cents_difference(f0[user_i], ref_f0[ref_i])
                for user_i, ref_i in zip(aligned.user_indices, aligned.ref_indices)
            ])
            cents_diff = np.array(cents_diff, dtype=np.float32)
            cents_diff = cents_diff[np.isfinite(cents_diff)]
            if len(cents_diff) >= 5:
                return self._score_from_cents_diff(cents_diff)

        return self._score_against_reference_naive(f0, voiced_flag, ref_f0)

    def _score_against_reference_naive(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        ref_f0: np.ndarray,
    ) -> float:
        """Fallback cũ: crop theo min length và so frame-by-frame."""
        min_len = min(len(f0), len(ref_f0))
        f0_crop = f0[:min_len]
        ref_crop = ref_f0[:min_len]
        voiced_crop = voiced_flag[:min_len]

        # Chỉ so sánh các frame có giọng
        mask = voiced_crop & (f0_crop > 0) & (ref_crop > 0)
        if np.sum(mask) < 5:
            return 60.0

        # Tính sai lệch cents
        cents_diff = np.abs([
            self.cents_difference(f0_crop[i], ref_crop[i])
            for i in range(min_len) if mask[i]
        ])

        return self._score_from_cents_diff(np.array(cents_diff, dtype=np.float32))

    @staticmethod
    def _score_from_cents_diff(cents_diff: np.ndarray) -> float:
        if len(cents_diff) == 0:
            return 60.0

        # Scoring dựa trên sai lệch cents
        # ≤ 25 cents: hoàn hảo, ≤ 50: tốt, ≤ 100: chấp nhận
        perfect = np.sum(cents_diff <= 25)
        good    = np.sum((cents_diff > 25) & (cents_diff <= 50))
        ok      = np.sum((cents_diff > 50) & (cents_diff <= 100))
        bad     = np.sum(cents_diff > 100)
        total   = len(cents_diff)

        score = (perfect * 100 + good * 75 + ok * 50 + bad * 10) / total
        return float(np.clip(score, 0, 100))

    def _score_internal_consistency(self, voiced_f0: np.ndarray) -> float:
        """
        Đánh giá tính nhất quán nội bộ khi không có bản nhạc gốc.
        Phương pháp: đo độ trơn tru của đường pitch (ít nhảy vọt = tốt hơn)
        """
        midi_vals = np.array([self.hz_to_midi(f) for f in voiced_f0 if f > 0])
        if len(midi_vals) < 2:
            return 60.0

        # Tính độ biến thiên (bước nhảy bán cung)
        diffs = np.abs(np.diff(midi_vals))

        # Lọc những bước nhảy nhỏ là rung bình thường
        significant_jumps = diffs[diffs > settings.PITCH_JUMP_THRESHOLD]

        if len(significant_jumps) == 0:
            return 88.0  # Giọng rất ổn định

        # Tính trung bình độ lớn các bước nhảy
        mean_jump = np.mean(significant_jumps)
        jump_ratio = len(significant_jumps) / len(diffs)

        # Scoring: ít nhảy vọt, bước nhảy nhỏ = điểm cao
        jump_score   = max(0, 100 - mean_jump * settings.PITCH_JUMP_PENALTY_SCALE)
        ratio_score  = max(0, 100 - jump_ratio * settings.PITCH_JUMP_RATIO_SCALE)
        return float(np.clip((jump_score * 0.6 + ratio_score * 0.4), 0, 100))

    # ─────────────────────────────────────────
    # Stability (Vibrato & Jitter)
    # ─────────────────────────────────────────

    def analyze_vibrato(
        self, voiced_f0: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Phân tích vibrato (rung giọng đẹp, có chủ đích).
        Returns: (rate_hz, extent_semitones)
        """
        if len(voiced_f0) < 20:
            return None, None

        # Tính độ biến thiên của MIDI values
        midi_vals = np.array([self.hz_to_midi(f) for f in voiced_f0 if f > 0])
        if len(midi_vals) < 20:
            return None, None

        # Detrend để lấy phần dao động
        detrended = midi_vals - np.convolve(midi_vals, np.ones(10) / 10, mode="same")

        # FFT để tìm tần số dao động
        fft_vals = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended), d=self.hop_length / self.sr)

        # Vibrato thường trong khoảng 4-8 Hz
        vibrato_range = (freqs >= 4) & (freqs <= 8)
        if not np.any(vibrato_range):
            return None, None

        vibrato_fft = fft_vals[vibrato_range]
        vibrato_freqs = freqs[vibrato_range]

        if vibrato_fft.max() < np.percentile(fft_vals, 80):
            return None, None  # Vibrato quá yếu, không rõ ràng

        rate_hz = float(vibrato_freqs[np.argmax(vibrato_fft)])
        extent = float(np.std(detrended) * 2)  # ≈ peak-to-peak semitones
        return rate_hz, extent

    def compute_jitter(self, voiced_f0: np.ndarray) -> float:
        """
        Tính Jitter - % biến động chu kỳ cơ bản.
        Jitter cao → giọng run, mất ổn định.
        """
        periods = np.array([1.0 / f for f in voiced_f0 if f > 0])
        if len(periods) < 3:
            return 0.0
        diffs = np.abs(np.diff(periods))
        jitter = np.mean(diffs) / np.mean(periods) * 100
        return float(np.clip(jitter, 0, 50))

    def compute_shimmer(self, y: np.ndarray, voiced_f0: np.ndarray) -> float:
        """
        Tính Shimmer - % biến động biên độ.
        Shimmer cao → giọng rung lắc không đều.
        """
        # Tính RMS từng frame ngắn
        frame_size = int(self.sr / np.mean(voiced_f0[voiced_f0 > 0])) if np.any(voiced_f0 > 0) else 512
        frame_size = max(64, min(frame_size, 2048))
        frames = librosa.util.frame(y, frame_length=frame_size, hop_length=frame_size // 2)
        amplitudes = np.sqrt(np.mean(frames ** 2, axis=0))
        amplitudes = amplitudes[amplitudes > 1e-6]
        if len(amplitudes) < 3:
            return 0.0
        diffs = np.abs(np.diff(amplitudes))
        shimmer = np.mean(diffs) / np.mean(amplitudes) * 100
        return float(np.clip(shimmer, 0, 50))

    # ─────────────────────────────────────────
    # Vocal Range Detection
    # ─────────────────────────────────────────

    def detect_vocal_range(self, voiced_f0: np.ndarray) -> VocalRange:
        """Xác định tầm âm giọng hát"""
        if len(voiced_f0) == 0:
            return VocalRange.UNKNOWN

        f0_clean = voiced_f0[voiced_f0 > 0]
        if len(f0_clean) == 0:
            return VocalRange.UNKNOWN

        median_hz = float(np.median(f0_clean))
        p10 = float(np.percentile(f0_clean, 10))
        p90 = float(np.percentile(f0_clean, 90))

        best_match = VocalRange.UNKNOWN
        best_score = float("inf")

        for vr, (lo, hi) in VOCAL_RANGE_BOUNDARIES.items():
            center = (lo + hi) / 2
            dist = abs(median_hz - center)
            overlap = min(p90, hi) - max(p10, lo)
            range_size = hi - lo
            overlap_ratio = max(0, overlap / range_size)
            score = dist - overlap_ratio * 200
            if score < best_score:
                best_score = score
                best_match = vr

        return best_match

    # ─────────────────────────────────────────
    # Main Analysis
    # ─────────────────────────────────────────

    def analyze(
        self,
        y: np.ndarray,
        voiced_ratio: float,
        reference_f0: Optional[np.ndarray] = None,
    ) -> PitchAnalysis:
        """Phân tích pitch hoàn chỉnh"""
        logger.info("Đang phân tích pitch...")

        f0, voiced_flag, voiced_probs = self.extract_pitch(y)
        voiced_f0 = f0[voiced_flag & (f0 > 0)]

        # Thông số cơ bản
        avg_hz = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        avg_note = self.hz_to_note(avg_hz)
        vocal_range = self.detect_vocal_range(voiced_f0)

        # Độ ổn định pitch (nghịch đảo của std)
        if len(voiced_f0) > 1:
            midi_vals = np.array([self.hz_to_midi(f) for f in voiced_f0])
            pitch_std = float(np.std(midi_vals))
            pitch_stability = float(np.clip(1.0 - pitch_std / 12.0, 0, 1))
        else:
            pitch_stability = 0.5

        # Accuracy score
        accuracy_score = self.score_pitch_accuracy(f0, voiced_flag, reference_f0)

        # Out-of-tune segments (nhảy > 100 cents so với trung bình cục bộ)
        out_of_tune = self._count_out_of_tune_segments(f0, voiced_flag)

        # % nốt đúng cao độ
        pitch_accuracy_percent = float(accuracy_score)

        # Jitter & Shimmer
        jitter = self.compute_jitter(voiced_f0)

        # Vibrato
        vibrato_rate, vibrato_extent = self.analyze_vibrato(voiced_f0)

        # Final pitch score
        stability_bonus = pitch_stability * 10
        jitter_penalty = min(settings.PITCH_JITTER_PENALTY_CAP, jitter * settings.PITCH_JITTER_PENALTY_SCALE)
        pitch_score = float(np.clip(accuracy_score + stability_bonus - jitter_penalty, 0, 100))

        logger.info(f"Pitch score: {pitch_score:.1f}, avg={avg_hz:.1f}Hz ({avg_note}), range={vocal_range.value}")

        return PitchAnalysis(
            score=round(pitch_score, 2),
            average_pitch_hz=round(avg_hz, 2),
            average_pitch_note=avg_note,
            pitch_accuracy_percent=round(pitch_accuracy_percent, 2),
            out_of_tune_segments=out_of_tune,
            pitch_stability=round(pitch_stability, 4),
            vocal_range=vocal_range,
            voiced_ratio=round(voiced_ratio, 4),
        ), f0, voiced_flag

    def _count_out_of_tune_segments(
        self, f0: np.ndarray, voiced_flag: np.ndarray, window: int = 20
    ) -> int:
        """Đếm số đoạn lạc giọng đáng kể"""
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) < window:
            return 0

        count = 0
        midi_vals = np.array([self.hz_to_midi(f) for f in voiced_f0 if f > 0])
        for i in range(window, len(midi_vals)):
            local_mean = np.mean(midi_vals[max(0, i - window):i])
            if abs(midi_vals[i] - local_mean) > 2.0:  # > 2 semitones
                count += 1

        # Gộp các frames liên tiếp thành "segments"
        return max(0, count // 5)  # ~5 frames/segment


pitch_analyzer = PitchAnalyzer()

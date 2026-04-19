"""
Scoring Engine & Feedback Generator
- Tổng hợp điểm từ tất cả analyzers
- Tạo phản hồi chi tiết bằng tiếng Việt
- Phân tích từng đoạn
"""
import numpy as np
import librosa
import time
import tempfile
import requests
from pathlib import Path
from typing import Optional, List

from app.core.config import settings
from app.core.logger import logger
from app.models.scoring import (
    ScoringResult, ScoreGrade, Feedback, SegmentScore,
    PitchAnalysis, RhythmAnalysis, StabilityAnalysis, DynamicsAnalysis,
)
from app.services.audio_preprocessor import preprocessor
from app.services.pitch_analyzer import pitch_analyzer
from app.services.rhythm_analyzer import rhythm_analyzer
from app.services.stability_dynamics_analyzer import stability_analyzer, dynamics_analyzer
from app.services.vocal_separator import vocal_separator
from app.services.storage_service import storage_service
from app.db.models import Song
from app.db.session import SessionLocal


# ─────────────────────────────────────────────
# Grade boundaries
# ─────────────────────────────────────────────
def score_to_grade(score: float) -> ScoreGrade:
    if score >= 90: return ScoreGrade.S
    if score >= 80: return ScoreGrade.A
    if score >= 70: return ScoreGrade.B
    if score >= 60: return ScoreGrade.C
    if score >= 50: return ScoreGrade.D
    return ScoreGrade.F


# ─────────────────────────────────────────────
# Feedback Generator
# ─────────────────────────────────────────────
class FeedbackGenerator:
    """Tạo phản hồi ngôn ngữ tự nhiên bằng tiếng Việt"""

    def generate(
        self,
        pitch: PitchAnalysis,
        rhythm: RhythmAnalysis,
        stability: StabilityAnalysis,
        dynamics: DynamicsAnalysis,
        total_score: float,
        grade: ScoreGrade,
    ) -> Feedback:
        strengths = []
        improvements = []
        tips = []

        # ── Pitch Feedback ──────────────────────
        if pitch.score >= 85:
            strengths.append(f"🎵 Cao độ rất chuẩn ({pitch.score:.0f}/100) - bạn hát đúng nốt rất tốt!")
        elif pitch.score >= 70:
            strengths.append(f"🎵 Cao độ khá tốt ({pitch.score:.0f}/100)")
        else:
            improvements.append(f"🎵 Cao độ cần cải thiện ({pitch.score:.0f}/100) - có {pitch.out_of_tune_segments} đoạn lạc giọng")
            tips.append("💡 Luyện tập với đàn piano hoặc app dò nốt để cải thiện độ chuẩn cao độ")

        if pitch.pitch_stability >= 0.8:
            strengths.append(f"📏 Giọng rất ổn định, ít dao động không cần thiết")
        elif pitch.pitch_stability < 0.5:
            improvements.append("📏 Giọng còn chưa ổn định, cao độ dao động nhiều")
            tips.append("💡 Tập hít thở sâu (diaphragmatic breathing) để kiểm soát hơi tốt hơn")

        # ── Vibrato Feedback ─────────────────────
        if stability.vibrato_rate_hz:
            if 4.5 <= stability.vibrato_rate_hz <= 7.5:
                strengths.append(f"🌀 Vibrato đẹp và tự nhiên ({stability.vibrato_rate_hz:.1f} Hz)")
            else:
                improvements.append(f"🌀 Vibrato chưa đều ({stability.vibrato_rate_hz:.1f} Hz, lý tưởng: 5-7 Hz)")

        # ── Rhythm Feedback ──────────────────────
        if rhythm.score >= 85:
            strengths.append(f"🥁 Giữ nhịp rất tốt ({rhythm.score:.0f}/100), tempo {rhythm.estimated_tempo_bpm:.0f} BPM ổn định")
        elif rhythm.score >= 70:
            strengths.append(f"🥁 Nhịp điệu khá ổn ({rhythm.score:.0f}/100)")
        else:
            improvements.append(f"🥁 Nhịp điệu chưa đều ({rhythm.score:.0f}/100), sai lệch trung bình {rhythm.rhythm_deviation_ms:.0f}ms")
            tips.append("💡 Tập hát kèm metronome để cải thiện cảm giác nhịp")

        # ── Stability Feedback ───────────────────
        if stability.tremolo_detected:
            improvements.append("⚠️ Phát hiện run giọng không đều (tremolo) - có thể do căng thẳng hoặc hơi thở không đều")
            tips.append("💡 Thư giãn cổ họng và luyện tập bài tập legato chậm")

        if stability.jitter_percent > 2.0:
            improvements.append(f"🔊 Jitter cao ({stability.jitter_percent:.1f}%) - chu kỳ âm thanh không đều")
        if stability.shimmer_percent > 5.0:
            improvements.append(f"📊 Shimmer cao ({stability.shimmer_percent:.1f}%) - biên độ dao động không đều")

        if stability.breathiness_score > 0.6:
            improvements.append("💨 Giọng hơi nhiều hơi thở - có thể thiếu hỗ trợ hơi thở")
            tips.append("💡 Luyện bài tập 'staccato' để tăng cường kiểm soát luồng hơi")

        # ── Dynamics Feedback ────────────────────
        if dynamics.score >= 80:
            strengths.append(f"🎭 Biểu cảm tốt ({dynamics.score:.0f}/100) - giọng hát có hồn và cảm xúc")
        elif dynamics.dynamic_range_db < 10:
            improvements.append("🎭 Giọng hát đơn điệu, thiếu biểu cảm - hãy thêm cảm xúc vào từng câu")
            tips.append("💡 Tập hát với các sắc thái piano (nhỏ) và forte (to) để tạo độ tương phản")
        elif dynamics.score < 60:
            improvements.append(f"🎭 Diễn đạt cảm xúc cần phong phú hơn ({dynamics.score:.0f}/100)")

        # ── Vocal Range ──────────────────────────
        strengths.append(f"🎤 Tầm âm giọng hát: {pitch.vocal_range.value} (cao độ trung bình: {pitch.average_pitch_note})")

        # ── Overall Comment ──────────────────────
        overall = self._overall_comment(grade, total_score, pitch, rhythm, stability, dynamics)

        # Đảm bảo luôn có tips
        if not tips:
            tips.append("💡 Tiếp tục luyện tập đều đặn mỗi ngày để duy trì và nâng cao giọng hát")
        if not improvements:
            improvements.append("🔄 Tiếp tục duy trì phong độ và thử thách với những bài hát khó hơn")

        return Feedback(
            strengths=strengths[:5],
            improvements=improvements[:5],
            tips=tips[:4],
            overall_comment=overall,
        )

    def _overall_comment(
        self,
        grade: ScoreGrade,
        score: float,
        pitch: PitchAnalysis,
        rhythm: RhythmAnalysis,
        stability: StabilityAnalysis,
        dynamics: DynamicsAnalysis,
    ) -> str:
        weakest = min(
            [("cao độ", pitch.score), ("nhịp điệu", rhythm.score),
             ("ổn định", stability.score), ("biểu cảm", dynamics.score)],
            key=lambda x: x[1]
        )
        strongest = max(
            [("cao độ", pitch.score), ("nhịp điệu", rhythm.score),
             ("ổn định", stability.score), ("biểu cảm", dynamics.score)],
            key=lambda x: x[1]
        )

        grade_comments = {
            ScoreGrade.S: f"🏆 Xuất sắc! {score:.1f} điểm - Giọng hát của bạn gần như hoàn hảo. Điểm mạnh nhất là {strongest[0]} ({strongest[1]:.0f}/100).",
            ScoreGrade.A: f"⭐ Giỏi! {score:.1f} điểm - Giọng hát rất tốt. Hãy tập trung cải thiện {weakest[0]} ({weakest[1]:.0f}/100) để đạt S rank.",
            ScoreGrade.B: f"👍 Khá! {score:.1f} điểm - Nền tảng tốt. Điểm yếu nhất cần cải thiện là {weakest[0]} ({weakest[1]:.0f}/100).",
            ScoreGrade.C: f"📚 Trung bình. {score:.1f} điểm - Bạn cần luyện tập thêm, đặc biệt về {weakest[0]}.",
            ScoreGrade.D: f"💪 Cần cố gắng hơn. {score:.1f} điểm - Đừng nản lòng, hãy luyện tập từng ngày!",
            ScoreGrade.F: f"🌱 Mới bắt đầu. {score:.1f} điểm - Mọi ca sĩ đều bắt đầu từ đây. Kiên trì luyện tập nhé!",
        }
        return grade_comments.get(grade, f"Tổng điểm: {score:.1f}/100")


# ─────────────────────────────────────────────
# Segment Analyzer
# ─────────────────────────────────────────────
class SegmentAnalyzer:
    """Phân tích điểm từng đoạn của bài hát"""

    def __init__(self, segment_duration: float = 5.0):
        self.segment_duration = segment_duration  # giây

    def analyze_segments(
        self,
        y: np.ndarray,
        sr: int,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
    ) -> List[SegmentScore]:
        """Chia bài hát thành đoạn và chấm điểm từng đoạn"""
        duration = len(y) / sr
        segments = []
        hop = settings.HOP_LENGTH

        num_segments = max(1, int(duration / self.segment_duration))
        segment_frames = len(f0) // num_segments

        for i in range(num_segments):
            start_frame = i * segment_frames
            end_frame = min((i + 1) * segment_frames, len(f0))

            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop)
            end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop)

            seg_f0 = f0[start_frame:end_frame]
            seg_voiced = voiced_flag[start_frame:end_frame]
            voiced_f0 = seg_f0[seg_voiced & (seg_f0 > 0)]

            if len(voiced_f0) < 3:
                pitch_score = 50.0
                feedback = "⚠️ Đoạn này có ít giọng hát"
            else:
                # Tính ổn định pitch của đoạn
                midi_vals = 69 + 12 * np.log2(np.maximum(voiced_f0, 1e-6) / 440)
                std_midi = float(np.std(midi_vals))
                stability = float(np.clip(1.0 - std_midi / 8.0, 0, 1))

                # Tính jumps
                jumps = np.abs(np.diff(midi_vals))
                big_jumps = np.sum(jumps > 1.5)
                jump_penalty = min(30, big_jumps * 5)

                pitch_score = float(np.clip(stability * 90 - jump_penalty, 0, 100))
                feedback = self._segment_feedback(pitch_score)

            segments.append(SegmentScore(
                start_time=round(float(start_time), 2),
                end_time=round(float(end_time), 2),
                pitch_score=round(pitch_score, 1),
                overall_score=round(pitch_score, 1),
                feedback=feedback,
            ))

        return segments

    def _segment_feedback(self, score: float) -> str:
        if score >= 85: return "✅ Tuyệt vời"
        if score >= 70: return "👍 Tốt"
        if score >= 55: return "👌 Ổn"
        if score >= 40: return "⚠️ Cần cải thiện"
        return "❌ Nhiều lỗi"


# ─────────────────────────────────────────────
# Main Scoring Engine
# ─────────────────────────────────────────────
class VocalScoringEngine:
    """Engine chấm điểm giọng hát chính"""

    def __init__(self):
        self.feedback_gen = FeedbackGenerator()
        self.segment_analyzer = SegmentAnalyzer()
        self._reference_f0_cache = {}
        self._reference_audio_cache = {}

    def _resolve_reference_file_id(self, song_id: Optional[str]) -> Optional[str]:
        if not song_id:
            return None
        db = SessionLocal()
        try:
            song = db.query(Song).filter(Song.id == song_id).first()
            if not song or not song.has_reference_audio:
                return None
            return song.file_id
        except Exception as exc:
            logger.warning(f"Không truy vấn được song reference trong DB: {exc}")
            return None
        finally:
            db.close()

    def _load_reference_f0(self, song_id: Optional[str]) -> Optional[np.ndarray]:
        if not song_id:
            return None
        if song_id in self._reference_f0_cache:
            return self._reference_f0_cache[song_id]

        try:
            ref_audio = self._load_reference_audio(song_id)
            if ref_audio is None:
                return None
            ref_f0, _, _ = pitch_analyzer.extract_pitch(ref_audio)
            self._reference_f0_cache[song_id] = ref_f0
            return ref_f0
        except Exception as exc:
            logger.warning(f"Không load được reference_f0 cho song_id={song_id}: {exc}")
            return None

    def _load_reference_audio(self, song_id: Optional[str]) -> Optional[np.ndarray]:
        if not song_id:
            return None
        if song_id in self._reference_audio_cache:
            return self._reference_audio_cache[song_id]

        file_id = self._resolve_reference_file_id(song_id)
        if not file_id:
            return None

        temp_path = None
        try:
            suffix = Path(file_id).suffix or ".wav"
            ref_bytes = storage_service.download_bytes(file_id)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(ref_bytes)
                temp_path = temp_audio.name

            ref_data = preprocessor.preprocess(temp_path)
            ref_y = ref_data["y_raw"]
            ref_sr = ref_data["sr"]

            if not settings.SKIP_REFERENCE_SEPARATION:
                ref_y = vocal_separator.separate(ref_y, ref_sr, source_path=temp_path)
            ref_y = preprocessor.normalize_audio(ref_y)
            ref_y = preprocessor.remove_silence(ref_y)

            self._reference_audio_cache[song_id] = ref_y
            return ref_y
        except Exception as exc:
            logger.warning(f"Không load được reference audio cho song_id={song_id}: {exc}")
            return None
        finally:
            try:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _load_reference_audio_from_url(self, url: str, song_id: Optional[str] = None) -> Optional[np.ndarray]:
        """Download reference audio từ URL (được gửi bởi Node.js backend)"""
        cache_key = song_id or url
        if cache_key in self._reference_audio_cache:
            return self._reference_audio_cache[cache_key]

        temp_path = None
        try:
            logger.info(f"Đang tải reference audio từ URL: {url}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Xác định extension từ URL hoặc content-type
            url_path = Path(url.split("?")[0])
            suffix = url_path.suffix or ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(resp.content)
                temp_path = temp_audio.name

            ref_data = preprocessor.preprocess(temp_path)
            ref_y = ref_data["y_raw"]
            ref_sr = ref_data["sr"]

            if not settings.SKIP_REFERENCE_SEPARATION:
                ref_y = vocal_separator.separate(ref_y, ref_sr, source_path=temp_path)
            ref_y = preprocessor.normalize_audio(ref_y)
            ref_y = preprocessor.remove_silence(ref_y)

            self._reference_audio_cache[cache_key] = ref_y
            logger.info(f"Đã tải thành công reference audio từ URL ({len(resp.content)} bytes)")
            return ref_y
        except Exception as exc:
            logger.warning(f"Không tải được reference audio từ URL {url}: {exc}")
            return None
        finally:
            try:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def score(
        self,
        audio_path: str,
        song_title: Optional[str] = None,
        song_id: Optional[str] = None,
        reference_audio_url: Optional[str] = None,
    ) -> ScoringResult:
        start_time = time.time()
        logger.info(f"{'='*50}")
        logger.info(f"Bắt đầu chấm điểm: {song_title or 'Unknown Song'}")

        # 1. Preprocessing
        data = preprocessor.preprocess(audio_path)
        y = data["y"]
        y_raw = data["y_raw"]
        sr = data["sr"]
        duration = data["duration"]
        voiced_ratio = data["voiced_ratio"]

        # 1.1 Optional vocal separation (feature flag)
        y = vocal_separator.separate(y_raw, sr, source_path=audio_path)
        y = preprocessor.normalize_audio(y)
        y = preprocessor.remove_silence(y)
        duration = len(y) / sr
        voiced_mask, _ = preprocessor.voice_activity_detection(y)
        voiced_ratio = float(np.sum(voiced_mask)) / len(voiced_mask) if len(voiced_mask) > 0 else 0.0

        # 1.2 Load reference audio: ưu tiên URL từ Node.js, fallback sang DB lookup
        reference_audio = None
        if reference_audio_url:
            reference_audio = self._load_reference_audio_from_url(reference_audio_url, song_id)
        if reference_audio is None:
            reference_audio = self._load_reference_audio(song_id)
        reference_f0 = self._load_reference_f0(song_id)

        # 2. Pitch Analysis
        pitch_result, f0, voiced_flag = pitch_analyzer.analyze(
            y, voiced_ratio, reference_f0=reference_f0
        )

        # Jitter & Shimmer (cần voiced_f0)
        voiced_f0 = f0[voiced_flag & (f0 > 0)]
        jitter = pitch_analyzer.compute_jitter(voiced_f0)
        shimmer = pitch_analyzer.compute_shimmer(y, voiced_f0)
        vibrato_rate, vibrato_extent = pitch_analyzer.analyze_vibrato(voiced_f0)

        # 3. Rhythm Analysis
        rhythm_result = rhythm_analyzer.analyze(y, reference_y=reference_audio)

        # 4. Stability Analysis
        stability_result = stability_analyzer.analyze(
            y, f0, voiced_flag, jitter, shimmer, vibrato_rate, vibrato_extent
        )

        # 5. Dynamics Analysis
        dynamics_result = dynamics_analyzer.analyze(y)

        # 6. Tổng điểm
        total_score = (
            pitch_result.score   * settings.WEIGHT_PITCH +
            rhythm_result.score  * settings.WEIGHT_RHYTHM +
            stability_result.score * settings.WEIGHT_STABILITY +
            dynamics_result.score  * settings.WEIGHT_DYNAMICS
        )
        total_score = float(np.clip(total_score, 0, 100))
        grade = score_to_grade(total_score)

        # 7. Segment analysis
        segments = self.segment_analyzer.analyze_segments(y, sr, f0, voiced_flag)

        # 8. Feedback
        feedback = self.feedback_gen.generate(
            pitch_result, rhythm_result, stability_result, dynamics_result,
            total_score, grade,
        )

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Hoàn tất! Tổng điểm: {total_score:.1f} ({grade.value}) | {processing_time:.0f}ms")

        return ScoringResult(
            song_title=song_title,
            audio_duration_seconds=round(duration, 2),
            sample_rate=sr,
            total_score=round(total_score, 2),
            grade=grade,
            pitch=pitch_result,
            rhythm=rhythm_result,
            stability=stability_result,
            dynamics=dynamics_result,
            segments=segments,
            feedback=feedback,
            processing_time_ms=round(processing_time, 2),
        )


scoring_engine = VocalScoringEngine()

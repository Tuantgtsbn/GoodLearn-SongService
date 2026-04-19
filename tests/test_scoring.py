"""
Test Suite - Kiểm tra toàn bộ pipeline chấm điểm
Chạy: python tests/test_scoring.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import time

from app.utils.audio_utils import generate_test_audio_sine, generate_test_audio_melody, get_audio_info
from app.core.config import settings
from app.services.audio_preprocessor import preprocessor
from app.services.pitch_analyzer import pitch_analyzer
from app.services.rhythm_analyzer import rhythm_analyzer
from app.services.stability_dynamics_analyzer import stability_analyzer, dynamics_analyzer
from app.services.scoring_engine import scoring_engine
from app.services.vocal_separator import vocal_separator


def print_separator(title: str = ""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def test_audio_generation():
    print_separator("TEST 1: Tạo audio test")
    
    # Tạo sine wave (nốt A4 = 440Hz)
    sine_path = "/tmp/test_sine_440.wav"
    generate_test_audio_sine(frequency=440.0, duration=5.0, output_path=sine_path)
    info = get_audio_info(sine_path)
    print(f"✅ Sine wave A4 (440Hz): {info}")

    # Tạo giai điệu Do Re Mi
    melody_path = "/tmp/test_melody.wav"
    generate_test_audio_melody(output_path=melody_path)
    info = get_audio_info(melody_path)
    print(f"✅ Giai điệu Do Re Mi: {info}")

    return sine_path, melody_path


def test_preprocessing(audio_path: str):
    print_separator("TEST 2: Audio Preprocessing")
    
    data = preprocessor.preprocess(audio_path)
    print(f"✅ Duration: {data['duration']:.2f}s")
    print(f"✅ Sample rate: {data['sr']}Hz")
    print(f"✅ Voiced ratio: {data['voiced_ratio']:.3f}")
    print(f"✅ Samples: {len(data['y'])}")
    return data


def test_pitch_analysis(audio_path: str):
    print_separator("TEST 3: Pitch Analysis")
    
    data = preprocessor.preprocess(audio_path)
    y = data["y"]
    voiced_ratio = data["voiced_ratio"]

    pitch_result, f0, voiced_flag = pitch_analyzer.analyze(y, voiced_ratio)

    print(f"✅ Pitch Score: {pitch_result.score:.2f}/100")
    print(f"✅ Average Pitch: {pitch_result.average_pitch_hz:.2f}Hz ({pitch_result.average_pitch_note})")
    print(f"✅ Vocal Range: {pitch_result.vocal_range.value}")
    print(f"✅ Pitch Stability: {pitch_result.pitch_stability:.4f}")
    print(f"✅ Out-of-tune segments: {pitch_result.out_of_tune_segments}")
    print(f"✅ Voiced Ratio: {pitch_result.voiced_ratio:.3f}")

    return pitch_result, f0, voiced_flag


def test_rhythm_analysis(audio_path: str):
    print_separator("TEST 4: Rhythm Analysis")
    
    data = preprocessor.preprocess(audio_path)
    y = data["y"]

    rhythm_result = rhythm_analyzer.analyze(y)

    print(f"✅ Rhythm Score: {rhythm_result.score:.2f}/100")
    print(f"✅ Tempo: {rhythm_result.estimated_tempo_bpm:.1f} BPM")
    print(f"✅ Beat Consistency: {rhythm_result.beat_consistency:.4f}")
    print(f"✅ Onset Regularity: {rhythm_result.onset_regularity:.4f}")
    print(f"✅ Rhythm Deviation: {rhythm_result.rhythm_deviation_ms:.1f}ms")

    return rhythm_result


def test_stability_dynamics(audio_path: str):
    print_separator("TEST 5: Stability & Dynamics Analysis")
    
    data = preprocessor.preprocess(audio_path)
    y = data["y"]

    pitch_result, f0, voiced_flag = pitch_analyzer.analyze(y, data["voiced_ratio"])
    voiced_f0 = f0[voiced_flag & (f0 > 0)]

    jitter = pitch_analyzer.compute_jitter(voiced_f0)
    shimmer = pitch_analyzer.compute_shimmer(y, voiced_f0)
    vibrato_rate, vibrato_extent = pitch_analyzer.analyze_vibrato(voiced_f0)

    stability_result = stability_analyzer.analyze(
        y, f0, voiced_flag, jitter, shimmer, vibrato_rate, vibrato_extent
    )
    dynamics_result = dynamics_analyzer.analyze(y)

    print(f"✅ Stability Score: {stability_result.score:.2f}/100")
    print(f"✅ Jitter: {stability_result.jitter_percent:.4f}%")
    print(f"✅ Shimmer: {stability_result.shimmer_percent:.4f}%")
    print(f"✅ Breathiness: {stability_result.breathiness_score:.4f}")
    print(f"✅ Tremolo: {stability_result.tremolo_detected}")
    if stability_result.vibrato_rate_hz:
        print(f"✅ Vibrato: {stability_result.vibrato_rate_hz:.1f}Hz, {stability_result.vibrato_extent_semitones:.2f} semitones")

    print(f"\n✅ Dynamics Score: {dynamics_result.score:.2f}/100")
    print(f"✅ Dynamic Range: {dynamics_result.dynamic_range_db:.1f}dB")
    print(f"✅ Expressiveness: {dynamics_result.emotional_expressiveness:.4f}")

    return stability_result, dynamics_result


def test_full_scoring(audio_path: str, song_title: str = "Test Song"):
    print_separator(f"TEST 6: Full Scoring Pipeline - '{song_title}'")
    
    start = time.time()
    result = scoring_engine.score(audio_path, song_title=song_title)
    elapsed = time.time() - start

    print(f"\n{'─'*40}")
    print(f"  🎵 Bài hát: {result.song_title}")
    print(f"  ⏱️  Thời lượng: {result.audio_duration_seconds:.2f}s")
    print(f"  ⚡ Thời gian xử lý: {result.processing_time_ms:.0f}ms")
    print(f"{'─'*40}")
    print(f"  🏆 TỔNG ĐIỂM: {result.total_score:.2f}/100  [{result.grade.value}]")
    print(f"{'─'*40}")
    print(f"  🎵 Cao độ:    {result.pitch.score:.1f}/100")
    print(f"  🥁 Nhịp điệu: {result.rhythm.score:.1f}/100")
    print(f"  📏 Ổn định:   {result.stability.score:.1f}/100")
    print(f"  🎭 Biểu cảm:  {result.dynamics.score:.1f}/100")
    print(f"{'─'*40}")

    print(f"\n  💪 Điểm mạnh:")
    for s in result.feedback.strengths:
        print(f"    {s}")

    print(f"\n  📝 Cần cải thiện:")
    for i in result.feedback.improvements:
        print(f"    {i}")

    print(f"\n  💡 Lời khuyên:")
    for t in result.feedback.tips:
        print(f"    {t}")

    print(f"\n  📢 Nhận xét: {result.feedback.overall_comment}")

    print(f"\n  📊 Phân tích từng đoạn ({len(result.segments)} đoạn):")
    for seg in result.segments:
        print(f"    [{seg.start_time:.1f}s - {seg.end_time:.1f}s] Score={seg.overall_score:.1f} | {seg.feedback}")

    return result


def test_dtw_alignment_reference_scoring():
    print_separator("TEST 8: DTW Alignment Reference Scoring")

    original_flag = settings.ENABLE_DTW_ALIGNMENT
    settings.ENABLE_DTW_ALIGNMENT = True
    try:
        # Same contour, but user has denser frames (tempo mismatch simulation).
        ref_f0 = np.array([220.0, 246.94, 261.63, 293.66, 329.63], dtype=np.float32)
        user_f0 = np.repeat(ref_f0, 2)
        voiced_flag = np.ones(len(user_f0), dtype=bool)

        score = pitch_analyzer.score_pitch_accuracy(user_f0, voiced_flag, ref_f0)
        print(f"✅ DTW reference score (tempo mismatch): {score:.2f}/100")
    finally:
        settings.ENABLE_DTW_ALIGNMENT = original_flag


def test_vocal_separator_fallback():
    print_separator("TEST 9: Vocal Separator Fallback")

    original_enable = settings.ENABLE_VOCAL_SEPARATION
    original_backend = settings.VOCAL_SEPARATOR_BACKEND

    settings.ENABLE_VOCAL_SEPARATION = True
    settings.VOCAL_SEPARATOR_BACKEND = "none"
    try:
        y = np.random.uniform(-0.2, 0.2, size=22050).astype(np.float32)
        y_out = vocal_separator.separate(y, sr=22050)
        print(f"✅ Separator fallback output length: {len(y_out)} samples")
    finally:
        settings.ENABLE_VOCAL_SEPARATION = original_enable
        settings.VOCAL_SEPARATOR_BACKEND = original_backend


def test_rhythm_reference_alignment():
    print_separator("TEST 10: Rhythm Reference Alignment")

    original_flag = settings.ENABLE_DTW_ALIGNMENT
    settings.ENABLE_DTW_ALIGNMENT = True
    try:
        # User vào trễ 0.8s nhưng giữ nhịp giống reference.
        ref_onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        user_onsets = ref_onsets + 0.8

        dev_ms = rhythm_analyzer.compute_reference_rhythm_deviation(user_onsets, ref_onsets)
        print(f"✅ Rhythm deviation with start offset: {dev_ms:.2f}ms")

        # Tempo mismatch nhẹ: user có thêm onset trung gian.
        user_dense = np.array([0.8, 1.1, 1.3, 1.6, 1.8, 2.1, 2.3, 2.6], dtype=np.float32)
        dev_dense_ms = rhythm_analyzer.compute_reference_rhythm_deviation(user_dense, ref_onsets)
        print(f"✅ Rhythm deviation with tempo mismatch: {dev_dense_ms:.2f}ms")
    finally:
        settings.ENABLE_DTW_ALIGNMENT = original_flag


def run_all_tests():
    print("\n" + "="*60)
    print("  🎤 VOCAL SCORING SYSTEM - TEST SUITE")
    print("="*60)

    try:
        # Test 1: Generate test audio
        sine_path, melody_path = test_audio_generation()

        # Test với sine wave (nốt đơn, dễ đạt điểm cao)
        test_preprocessing(sine_path)
        test_pitch_analysis(sine_path)
        test_rhythm_analysis(sine_path)
        test_stability_dynamics(sine_path)
        
        # Full scoring với sine wave
        result_sine = test_full_scoring(sine_path, "Nốt A4 (440Hz) - Sine Wave")

        # Full scoring với giai điệu
        print_separator("TEST 7: Full Scoring - Giai điệu Do Re Mi")
        result_melody = test_full_scoring(melody_path, "Giai điệu Do Re Mi")

        # New feature checks
        test_dtw_alignment_reference_scoring()
        test_vocal_separator_fallback()
        test_rhythm_reference_alignment()

        print_separator("✅ TẤT CẢ TESTS PASSED!")
        print(f"  Sine wave score: {result_sine.total_score:.1f} ({result_sine.grade.value})")
        print(f"  Melody score:    {result_melody.total_score:.1f} ({result_melody.grade.value})")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

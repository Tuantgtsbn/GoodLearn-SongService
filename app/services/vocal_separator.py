"""
Vocal Separation Service
- Optional vocal isolation before analysis
- Environment-aware backend selection with safe fallback
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import importlib
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from app.core.config import settings
from app.core.logger import logger


class VocalSeparator:
    """Isolate vocal signal from mixed audio when feature flag is enabled."""

    def __init__(self) -> None:
        self._demucs_unavailable = False

    def separate(
        self,
        y: np.ndarray,
        sr: int,
        source_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return a vocal-enhanced waveform.
        Falls back to the original waveform on any failure.
        """
        if not settings.ENABLE_VOCAL_SEPARATION:
            return y

        duration_seconds = len(y) / float(sr) if sr > 0 else 0.0
        if duration_seconds > settings.VOCAL_SEPARATOR_MAX_DURATION_SECONDS:
            logger.info(
                "Skip vocal separation: audio too long %.1fs > %.1fs",
                duration_seconds,
                settings.VOCAL_SEPARATOR_MAX_DURATION_SECONDS,
            )
            return y

        backend = self._select_backend(source_path)
        device = (settings.VOCAL_SEPARATOR_DEVICE or "auto").lower()
        if backend == "demucs":
            return self._separate_with_demucs(y, sr, source_path, device)
        if backend == "spleeter":
            return self._separate_with_spleeter(y)

        logger.warning("Vocal separation backend unavailable. Using original waveform.")
        return y

    def _select_backend(self, source_path: Optional[str]) -> str:
        backend = (settings.VOCAL_SEPARATOR_BACKEND or "auto").lower()
        if backend == "none":
            return "none"
        if backend == "demucs":
            return "demucs" if self._can_use_demucs(source_path) else "none"
        if backend == "spleeter":
            return "spleeter" if self._can_use_spleeter() else "none"

        # auto mode
        if self._can_use_demucs(source_path):
            return "demucs"
        if self._can_use_spleeter():
            return "spleeter"
        return "none"

    @staticmethod
    def _can_use_spleeter() -> bool:
        try:
            return importlib.util.find_spec("spleeter.separator") is not None
        except Exception:
            return False

    @staticmethod
    def _runtime_can_use_torchcodec() -> bool:
        try:
            importlib.import_module("torchaudio")
            importlib.import_module("torchcodec")
            return True
        except Exception:
            return False

    def _can_use_demucs(self, source_path: Optional[str]) -> bool:
        if self._demucs_unavailable:
            return False
        if not source_path:
            return False
        if shutil.which("demucs"):
            return self._runtime_can_use_torchcodec()
        has_demucs = importlib.util.find_spec("demucs") is not None
        return has_demucs and self._runtime_can_use_torchcodec()

    def _separate_with_spleeter(self, y: np.ndarray) -> np.ndarray:
        try:
            separator_module = importlib.import_module("spleeter.separator")
            Separator = getattr(separator_module, "Separator")

            separator = Separator("spleeter:2stems")
            waveform = np.expand_dims(y, axis=1)
            prediction = separator.separate(waveform)
            vocals = prediction.get("vocals")
            if vocals is None or vocals.size == 0:
                logger.warning("Spleeter returned empty vocals. Using original waveform.")
                return y

            if vocals.ndim == 2:
                vocals = vocals.mean(axis=1)
            vocals = vocals.astype(np.float32)
            peak = np.max(np.abs(vocals))
            if peak > 0:
                vocals = vocals / peak * 0.9
            logger.info("Vocal separation backend: spleeter")
            return vocals
        except Exception as exc:
            logger.warning(f"Spleeter separation failed: {exc}")
            return y

    def _separate_with_demucs(
        self,
        y: np.ndarray,
        sr: int,
        source_path: Optional[str],
        device: str,
    ) -> np.ndarray:
        if not source_path:
            logger.warning("Demucs requires source_path. Using original waveform.")
            return y

        stem_name = Path(source_path).stem
        with tempfile.TemporaryDirectory(prefix="demucs_sep_") as tmp_dir:
            try:
                output_dir = Path(tmp_dir) / "out"
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    "-m",
                    "demucs.separate",
                    "--two-stems=vocals",
                    "-o",
                    str(output_dir),
                    source_path,
                ]
                if device == "cpu":
                    cmd.insert(3, "--device")
                    cmd.insert(4, "cpu")
                elif device == "cuda":
                    cmd.insert(3, "--device")
                    cmd.insert(4, "cuda")

                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=max(5, int(settings.DEMUCS_TIMEOUT_SECONDS)),
                )
                if completed.returncode != 0:
                    logger.warning(f"Demucs separation failed: {completed.stderr.strip()}")
                    self._demucs_unavailable = True
                    return y

                vocals_path = self._find_demucs_vocals(output_dir, stem_name)
                if not vocals_path:
                    logger.warning("Demucs output vocals.wav not found. Using original waveform.")
                    return y

                vocals, out_sr = librosa.load(str(vocals_path), sr=sr, mono=True, dtype=np.float32)
                if len(vocals) == 0:
                    return y
                logger.info("Vocal separation backend: demucs")
                return vocals
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Demucs separation timeout after %ss. Using original waveform.",
                    settings.DEMUCS_TIMEOUT_SECONDS,
                )
                self._demucs_unavailable = True
                return y
            except Exception as exc:
                logger.warning(f"Demucs separation error: {exc}")
                self._demucs_unavailable = True
                return y

    @staticmethod
    def _find_demucs_vocals(output_dir: Path, stem_name: str) -> Optional[Path]:
        direct = output_dir / stem_name / "vocals.wav"
        if direct.exists():
            return direct

        for vocals_file in output_dir.glob(f"**/{stem_name}/vocals.wav"):
            return vocals_file
        return None


vocal_separator = VocalSeparator()

"""
Time Alignment Service
- DTW alignment for user/reference pitch tracks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from app.core.config import settings
from app.core.logger import logger


@dataclass
class AlignmentResult:
    user_indices: np.ndarray
    ref_indices: np.ndarray
    path_length: int


class TimeAligner:
    """Align pitch trajectories using Dynamic Time Warping."""

    def __init__(self) -> None:
        self.max_frames = max(100, int(settings.DTW_MAX_FRAMES))

    @staticmethod
    def _to_cents(values_hz: np.ndarray) -> np.ndarray:
        return 1200.0 * np.log2(np.maximum(values_hz, 1e-6) / 440.0)

    def _downsample_indices(self, indices: np.ndarray) -> np.ndarray:
        if len(indices) <= self.max_frames:
            return indices
        picked = np.linspace(0, len(indices) - 1, self.max_frames)
        return indices[np.round(picked).astype(int)]

    def align_pitch_tracks(
        self,
        user_f0: np.ndarray,
        user_voiced: np.ndarray,
        ref_f0: np.ndarray,
    ) -> Optional[AlignmentResult]:
        if not settings.ENABLE_DTW_ALIGNMENT:
            return None

        user_idx = np.where((user_voiced > 0) & (user_f0 > 0))[0]
        ref_idx = np.where(ref_f0 > 0)[0]

        if len(user_idx) < 5 or len(ref_idx) < 5:
            return None

        user_idx = self._downsample_indices(user_idx)
        ref_idx = self._downsample_indices(ref_idx)

        user_vals = self._to_cents(user_f0[user_idx])
        ref_vals = self._to_cents(ref_f0[ref_idx])

        if len(user_vals) == 0 or len(ref_vals) == 0:
            return None

        # Absolute cents distance matrix.
        cost_matrix = np.abs(user_vals[:, None] - ref_vals[None, :]).astype(np.float32)

        try:
            _, wp = librosa.sequence.dtw(C=cost_matrix)
            wp = np.asarray(wp)[::-1]
            aligned_user = user_idx[wp[:, 0]]
            aligned_ref = ref_idx[wp[:, 1]]

            logger.debug(
                "DTW alignment done: user_frames=%d, ref_frames=%d, path=%d",
                len(user_idx),
                len(ref_idx),
                len(wp),
            )
            return AlignmentResult(
                user_indices=aligned_user,
                ref_indices=aligned_ref,
                path_length=len(wp),
            )
        except Exception as exc:
            logger.warning(f"DTW alignment failed: {exc}")
            return None


time_aligner = TimeAligner()

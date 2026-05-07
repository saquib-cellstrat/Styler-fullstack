from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_landmarks(
    points: NDArray[np.float32], image_shape: tuple[int, int]
) -> NDArray[np.float32]:
    h, w = image_shape
    if h <= 0 or w <= 0:
        return points.astype(np.float32)
    norm = points.astype(np.float32).copy()
    norm[:, 0] /= float(max(w - 1, 1))
    norm[:, 1] /= float(max(h - 1, 1))
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def estimate_roll_from_eyes(points: NDArray[np.float32]) -> float:
    left_eye = points[36:42].mean(axis=0) if points.shape[0] >= 42 else points[0]
    right_eye = points[42:48].mean(axis=0) if points.shape[0] >= 48 else points[1]
    vec = right_eye - left_eye
    return float(np.degrees(np.arctan2(float(vec[1]), float(vec[0]))))

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def render_landmarks_overlay(
    rgb: NDArray[np.uint8], points: NDArray[np.float32], color: tuple[int, int, int] = (0, 255, 0)
) -> NDArray[np.uint8]:
    overlay = rgb.copy()
    for x, y in points:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, color, -1)
    return overlay

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class OptionalDepthEstimator:
    """CPU-friendly deterministic fallback depth estimator."""

    def estimate(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        v_bias = np.linspace(0.0, 1.0, gray.shape[0], dtype=np.float32)[:, None]
        depth = np.clip(0.7 * v_bias + 0.3 * (1.0 - np.abs(grad_y)), 0.0, 1.0)
        return depth.astype(np.float32)

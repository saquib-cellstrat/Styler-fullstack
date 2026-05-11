from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DebugExporter:
    def __init__(self, enabled: bool, root_dir: str) -> None:
        self._enabled = enabled
        self._root_dir = Path(root_dir)

    def write_rgb(self, name: str, image: np.ndarray) -> Path | None:
        if not self._enabled:
            return None
        self._root_dir.mkdir(parents=True, exist_ok=True)
        out = self._root_dir / f"{name}.png"
        rgb = np.asarray(image, dtype=np.uint8)
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return out

    def write_mask(self, name: str, mask: np.ndarray) -> Path | None:
        if not self._enabled:
            return None
        self._root_dir.mkdir(parents=True, exist_ok=True)
        out = self._root_dir / f"{name}.png"
        alpha = np.clip(mask, 0.0, 1.0)
        cv2.imwrite(str(out), (alpha * 255.0).astype(np.uint8))
        return out

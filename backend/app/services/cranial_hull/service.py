from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipeline.contracts import CranialHull, DenseLandmarks


class CranialHullEstimator:
    def estimate(self, image_shape: tuple[int, int], dense: DenseLandmarks) -> CranialHull:
        h, w = image_shape
        jaw = dense.jaw_contour
        left_temple = np.asarray(dense.temple_left, dtype=np.float32)
        right_temple = np.asarray(dense.temple_right, dtype=np.float32)
        temple_width = float(np.linalg.norm(right_temple - left_temple)) or 1.0
        eye_center = dense.points[36:48].mean(axis=0)
        crown = (eye_center + np.array([0.0, -1.15 * temple_width], dtype=np.float32)).astype(np.float32)
        crown[0] = np.clip(crown[0], 0, w - 1)
        crown[1] = np.clip(crown[1], 0, h - 1)

        axes = (max(int(round(temple_width * 0.72)), 1), max(int(round(temple_width * 0.95)), 1))
        center = (int(round(eye_center[0])), int(round((crown[1] + jaw[:, 1].mean()) * 0.5)))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, dense.pose.roll_deg, 190, 350, color=1.0, thickness=-1)
        arc = self._arc_points(np.array(center, dtype=np.float32), axes, dense.pose.roll_deg)
        ellipse = self._ellipse_points(np.array(center, dtype=np.float32), axes, dense.pose.roll_deg)
        anchors = np.vstack([arc[:: max(1, len(arc) // 6)], jaw[[0, 4, 8, 12, 16]]]).astype(np.float32)
        forehead_curvature = float(1.0 / max(axes[0], 1))
        return CranialHull(
            hull_mask=np.clip(mask, 0.0, 1.0).astype(np.float32),
            scalp_arc=arc,
            skull_ellipse=ellipse,
            crown_point=(float(crown[0]), float(crown[1])),
            forehead_curvature=forehead_curvature,
            canonical_anchors=anchors,
            metadata={"temple_width": temple_width, "crown_height_px": float(eye_center[1] - crown[1])},
        )

    def _arc_points(
        self, center: NDArray[np.float32], axes: tuple[int, int], roll_deg: float
    ) -> NDArray[np.float32]:
        theta = np.linspace(np.deg2rad(205), np.deg2rad(335), 28)
        return self._ellipse_param(center, axes, roll_deg, theta)

    def _ellipse_points(
        self, center: NDArray[np.float32], axes: tuple[int, int], roll_deg: float
    ) -> NDArray[np.float32]:
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        return self._ellipse_param(center, axes, roll_deg, theta)

    def _ellipse_param(
        self,
        center: NDArray[np.float32],
        axes: tuple[int, int],
        roll_deg: float,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        x = np.cos(theta) * axes[0]
        y = np.sin(theta) * axes[1]
        roll = np.deg2rad(roll_deg)
        xr = x * np.cos(roll) - y * np.sin(roll)
        yr = x * np.sin(roll) + y * np.cos(roll)
        pts = np.stack([center[0] + xr, center[1] + yr], axis=1)
        return pts.astype(np.float32)

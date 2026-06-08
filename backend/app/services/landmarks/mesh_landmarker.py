"""Dense facial mesh detector producing 478 anatomical points + head pose.

Wraps the bundled FaceLandmarker task model. Runs on CPU, detects its own
face (independent of the upstream 5-point detector), and is loaded once per
process because construction is expensive and the handle is reusable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipeline.contracts import HeadPose


@dataclass(frozen=True)
class MeshResult:
    points: NDArray[np.float32]  # (478, 2) pixel coordinates
    pose: HeadPose


class MeshLandmarker:
    """Lazy, process-wide singleton around the dense mesh task model."""

    _instances: dict[str, "MeshLandmarker"] = {}
    _class_lock = threading.Lock()

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._lock = threading.Lock()
        self._detector = None  # built on first use

    @classmethod
    def get(cls, model_path: str) -> "MeshLandmarker":
        with cls._class_lock:
            inst = cls._instances.get(model_path)
            if inst is None:
                inst = cls(model_path)
                cls._instances[model_path] = inst
            return inst

    def _ensure_detector(self):
        if self._detector is not None:
            return self._detector
        with self._lock:
            if self._detector is None:
                from mediapipe import Image, ImageFormat
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                self._Image = Image
                self._ImageFormat = ImageFormat
                base = mp_python.BaseOptions(model_asset_path=self._model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base,
                    num_faces=1,
                    output_facial_transformation_matrixes=True,
                    output_face_blendshapes=False,
                )
                self._detector = vision.FaceLandmarker.create_from_options(options)
        return self._detector

    def detect(self, rgb: NDArray[np.uint8]) -> MeshResult | None:
        detector = self._ensure_detector()
        h, w = rgb.shape[:2]
        mp_image = self._Image(
            image_format=self._ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        with self._lock:
            result = detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        points = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32
        )
        pose = self._pose_from_matrix(result)
        return MeshResult(points=points, pose=pose)

    @staticmethod
    def _pose_from_matrix(result) -> HeadPose:
        mats = getattr(result, "facial_transformation_matrixes", None)
        if not mats:
            return HeadPose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
        rot = np.array(mats[0], dtype=np.float64)[:3, :3]
        # Decompose to Euler (degrees). Sign convention matches image axes.
        sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
            yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
            roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
            yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
            roll = 0.0
        return HeadPose(yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=float(roll))


def landmarks_to_gray_for_debug(points: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.uint8]:
    canvas = np.zeros(shape, dtype=np.uint8)
    for x, y in points.astype(int):
        cv2.circle(canvas, (int(x), int(y)), 1, 255, -1)
    return canvas

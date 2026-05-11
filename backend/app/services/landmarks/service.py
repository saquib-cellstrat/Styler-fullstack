from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.pipeline.contracts import DenseLandmarks, HeadPose
from app.services.landmarks.normalization import estimate_roll_from_eyes, normalize_landmarks


class DenseLandmarkService:
    """Deterministic dense landmark generator with ONNX-ready interface."""

    def estimate(
        self,
        rgb: NDArray[np.uint8],
        face_box: tuple[float, float, float, float],
        face_landmarks_5: NDArray[np.float32],
    ) -> DenseLandmarks:
        points = self._heuristic_68(face_box, face_landmarks_5, rgb.shape[:2])
        normalized = normalize_landmarks(points, rgb.shape[:2])
        jaw = points[:17].astype(np.float32)
        forehead = self._forehead_from_brow(points)
        left_temple = tuple(np.asarray(forehead[0], dtype=np.float32))
        right_temple = tuple(np.asarray(forehead[-1], dtype=np.float32))
        pose = self._estimate_pose(points, face_landmarks_5)
        return DenseLandmarks(
            points=points.astype(np.float32),
            normalized_points=normalized,
            forehead_contour=forehead,
            jaw_contour=jaw,
            temple_left=left_temple,
            temple_right=right_temple,
            ear_left=tuple(np.asarray(points[0], dtype=np.float32)),
            ear_right=tuple(np.asarray(points[16], dtype=np.float32)),
            pose=pose,
            metadata={"landmark_count": float(points.shape[0]), "backend": "heuristic68"},
        )

    def save_json(self, landmarks: DenseLandmarks, path: str) -> None:
        payload = asdict(landmarks)
        payload["points"] = landmarks.points.tolist()
        payload["normalized_points"] = landmarks.normalized_points.tolist()
        payload["forehead_contour"] = landmarks.forehead_contour.tolist()
        payload["jaw_contour"] = landmarks.jaw_contour.tolist()
        payload = self._to_jsonable(payload)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _to_jsonable(self, value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        return value

    def _estimate_pose(
        self, points68: NDArray[np.float32], points5: NDArray[np.float32]
    ) -> HeadPose:
        roll = estimate_roll_from_eyes(points68)
        left_eye = points5[0]
        right_eye = points5[1]
        nose = points5[2]
        eye_mid = (left_eye + right_eye) / 2.0
        inter_eye = float(np.linalg.norm(right_eye - left_eye)) or 1.0
        yaw = float(np.degrees(np.arctan2(float(nose[0] - eye_mid[0]), inter_eye * 0.5)))
        brow_y = float(points68[19:25, 1].mean())
        chin_y = float(points68[8, 1])
        pitch = float(np.degrees(np.arctan2((chin_y - nose[1]) - (nose[1] - brow_y), inter_eye)))
        return HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

    def _forehead_from_brow(self, points68: NDArray[np.float32]) -> NDArray[np.float32]:
        brow = points68[17:27]
        offset = np.array([0.0, -0.18 * (points68[8, 1] - brow[:, 1].mean())], dtype=np.float32)
        return (brow + offset).astype(np.float32)

    def _heuristic_68(
        self,
        face_box: tuple[float, float, float, float],
        points5: NDArray[np.float32],
        image_shape: tuple[int, int],
    ) -> NDArray[np.float32]:
        x1, y1, x2, y2 = face_box
        h, w = image_shape
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        pts = np.zeros((68, 2), dtype=np.float32)
        t = np.linspace(np.pi * 0.05, np.pi * 0.95, 17)
        pts[:17, 0] = cx + np.cos(t) * (fw * 0.55)
        pts[:17, 1] = cy + np.sin(t) * (fh * 0.62)
        pts[17:22] = np.array(
            [[x1 + fw * 0.20 + i * fw * 0.08, y1 + fh * 0.30 - (i % 2) * fh * 0.03] for i in range(5)],
            dtype=np.float32,
        )
        pts[22:27] = np.array(
            [[x1 + fw * 0.52 + i * fw * 0.08, y1 + fh * 0.27 + (i % 2) * fh * 0.03] for i in range(5)],
            dtype=np.float32,
        )
        pts[27:31] = np.array([[cx, y1 + fh * (0.36 + i * 0.1)] for i in range(4)], dtype=np.float32)
        pts[31:36] = np.array(
            [[cx + fw * o, y1 + fh * y] for o, y in [(-0.12, 0.70), (-0.06, 0.74), (0.0, 0.76), (0.06, 0.74), (0.12, 0.70)]],
            dtype=np.float32,
        )
        left_eye = points5[0]
        right_eye = points5[1]
        eye_r = fw * 0.06
        le = np.array([[left_eye[0] + np.cos(a) * eye_r, left_eye[1] + np.sin(a) * eye_r * 0.7] for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)], dtype=np.float32)
        re = np.array([[right_eye[0] + np.cos(a) * eye_r, right_eye[1] + np.sin(a) * eye_r * 0.7] for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)], dtype=np.float32)
        pts[36:42] = le
        pts[42:48] = re
        mouth_c = (points5[3] + points5[4]) / 2.0
        mw = fw * 0.26
        mh = fh * 0.10
        outer = np.array([[mouth_c[0] + np.cos(a) * mw * 0.5, mouth_c[1] + np.sin(a) * mh] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)], dtype=np.float32)
        inner = np.array([[mouth_c[0] + np.cos(a) * mw * 0.28, mouth_c[1] + np.sin(a) * mh * 0.45] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)], dtype=np.float32)
        pts[48:60] = outer
        pts[60:68] = inner
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

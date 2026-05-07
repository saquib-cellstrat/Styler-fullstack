"""Frontal-face quality gate built on a RetinaFace ONNX model.

Implements the standard PyTorch_Retinaface decoding pipeline (prior-box
anchors + bbox/landmark variances + NMS) and a lightweight yaw heuristic
derived from the 5 facial landmarks. No heavy pose estimation models.
"""

from dataclasses import dataclass
from itertools import product
from math import ceil, degrees, atan2

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.model_registry import LoadedOnnxModel
from app.core.settings import Settings


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[float, float, float, float]
    score: float
    landmarks: NDArray[np.float32]
    estimated_yaw_deg: float


class NonFrontalFaceError(ValueError):
    """Raised when the donor image fails the frontal-face quality gate."""


class NoFaceFoundError(ValueError):
    """Raised when no face passes the detector confidence threshold."""


class FrontalFaceGate:
    """Detects faces with RetinaFace ONNX and rejects non-frontal donors."""

    def __init__(self, model: LoadedOnnxModel, settings: Settings) -> None:
        self.model = model
        self.settings = settings

    def evaluate(self, rgb: NDArray[np.uint8]) -> FaceDetection:
        size = self.settings.retinaface_input_size
        prepared, scale, pad_left, pad_top = self._letterbox_bgr(rgb, size)
        blob = self._preprocess(prepared)

        outputs = self.model.session.run(
            list(self.model.output_names),
            {self.model.input_names[0]: blob},
        )
        loc, conf, landms = self._align_outputs(outputs)

        priors = self._generate_priors(size)
        boxes = self._decode_boxes(loc[0], priors)
        landmarks = self._decode_landmarks(landms[0], priors)
        scores = self._softmax_face_score(conf[0])

        keep = scores >= self.settings.retinaface_conf_threshold
        if not np.any(keep):
            raise NoFaceFoundError("No face detected above confidence threshold")

        boxes = boxes[keep] * size
        landmarks = landmarks[keep] * size
        scores = scores[keep]

        keep_idx = self._nms(boxes, scores, self.settings.retinaface_nms_threshold)
        boxes = boxes[keep_idx]
        landmarks = landmarks[keep_idx]
        scores = scores[keep_idx]

        best = int(np.argmax(scores))
        box = boxes[best]
        lm = landmarks[best].reshape(5, 2)

        # Map coordinates from the letterbox back to the original image
        box = self._inverse_letterbox_box(box, scale, pad_left, pad_top)
        lm = self._inverse_letterbox_points(lm, scale, pad_left, pad_top)

        original_h, original_w = rgb.shape[:2]
        inter_eye = float(np.linalg.norm(lm[0] - lm[1]))
        if inter_eye / max(original_w, 1) < self.settings.min_inter_eye_fraction:
            raise NonFrontalFaceError("Face too small for high-fidelity extraction")

        yaw = self._estimate_yaw_deg(lm)
        if abs(yaw) > self.settings.max_abs_yaw_deg:
            raise NonFrontalFaceError(
                f"Face is not frontal enough (estimated yaw {yaw:.1f} deg)"
            )

        return FaceDetection(
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            score=float(scores[best]),
            landmarks=lm.astype(np.float32),
            estimated_yaw_deg=yaw,
        )

    @staticmethod
    def _letterbox_bgr(
        rgb: NDArray[np.uint8], size: int
    ) -> tuple[NDArray[np.uint8], float, int, int]:
        h, w = rgb.shape[:2]
        scale = min(size / w, size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        pad_left = (size - new_w) // 2
        pad_top = (size - new_h) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas, scale, pad_left, pad_top

    @staticmethod
    def _preprocess(rgb_canvas: NDArray[np.uint8]) -> NDArray[np.float32]:
        bgr = cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR).astype(np.float32)
        bgr -= np.asarray([104.0, 117.0, 123.0], dtype=np.float32)
        chw = np.transpose(bgr, (2, 0, 1))
        return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)

    @staticmethod
    def _align_outputs(
        outputs: list[NDArray[np.float32]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        # PyTorch_Retinaface returns (loc, conf, landms); detect by last-dim size.
        loc = next(o for o in outputs if o.shape[-1] == 4)
        landms = next(o for o in outputs if o.shape[-1] == 10)
        conf = next(o for o in outputs if o.shape[-1] == 2)
        return loc, conf, landms

    def _generate_priors(self, image_size: int) -> NDArray[np.float32]:
        feature_maps = [
            (ceil(image_size / step), ceil(image_size / step))
            for step in self.settings.retinaface_steps
        ]
        anchors: list[list[float]] = []
        for k, (fh, fw) in enumerate(feature_maps):
            min_sizes = self.settings.retinaface_min_sizes[k]
            step = self.settings.retinaface_steps[k]
            for i, j in product(range(fh), range(fw)):
                for min_size in min_sizes:
                    s_kx = min_size / image_size
                    s_ky = min_size / image_size
                    cx = (j + 0.5) * step / image_size
                    cy = (i + 0.5) * step / image_size
                    anchors.append([cx, cy, s_kx, s_ky])
        return np.asarray(anchors, dtype=np.float32)

    def _decode_boxes(
        self, loc: NDArray[np.float32], priors: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        v0, v1 = self.settings.retinaface_variances
        cxcy = priors[:, :2] + loc[:, :2] * v0 * priors[:, 2:]
        wh = priors[:, 2:] * np.exp(loc[:, 2:] * v1)
        boxes = np.empty_like(loc)
        boxes[:, 0] = cxcy[:, 0] - wh[:, 0] / 2
        boxes[:, 1] = cxcy[:, 1] - wh[:, 1] / 2
        boxes[:, 2] = cxcy[:, 0] + wh[:, 0] / 2
        boxes[:, 3] = cxcy[:, 1] + wh[:, 1] / 2
        return boxes

    def _decode_landmarks(
        self, landms: NDArray[np.float32], priors: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        v0, _ = self.settings.retinaface_land_variances
        decoded = np.empty_like(landms)
        for i in range(5):
            decoded[:, 2 * i] = priors[:, 0] + landms[:, 2 * i] * v0 * priors[:, 2]
            decoded[:, 2 * i + 1] = (
                priors[:, 1] + landms[:, 2 * i + 1] * v0 * priors[:, 3]
            )
        return decoded

    @staticmethod
    def _softmax_face_score(conf: NDArray[np.float32]) -> NDArray[np.float32]:
        exps = np.exp(conf - conf.max(axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return probs[:, 1].astype(np.float32)

    @staticmethod
    def _nms(
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        iou_threshold: float,
    ) -> NDArray[np.int64]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
            order = order[1:][iou <= iou_threshold]
        return np.asarray(keep, dtype=np.int64)

    @staticmethod
    def _inverse_letterbox_box(
        box: NDArray[np.float32], scale: float, pad_left: int, pad_top: int
    ) -> NDArray[np.float32]:
        out = box.copy()
        out[0::2] = (out[0::2] - pad_left) / scale
        out[1::2] = (out[1::2] - pad_top) / scale
        return out

    @staticmethod
    def _inverse_letterbox_points(
        points: NDArray[np.float32], scale: float, pad_left: int, pad_top: int
    ) -> NDArray[np.float32]:
        out = points.copy()
        out[:, 0] = (out[:, 0] - pad_left) / scale
        out[:, 1] = (out[:, 1] - pad_top) / scale
        return out

    @staticmethod
    def _estimate_yaw_deg(landmarks: NDArray[np.float32]) -> float:
        """Cheap symmetry-based yaw estimate.

        Uses the horizontal offset of the nose tip from the eye-midpoint,
        normalized by inter-eye distance, then mapped to an approximate
        angle. This is intentionally heuristic - good enough to reject
        clearly-turned faces without bringing in a full pose estimator.
        """
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        eye_mid = (left_eye + right_eye) / 2.0
        eye_vector = right_eye - left_eye
        inter_eye = float(np.linalg.norm(eye_vector)) or 1.0
        nose_offset_x = float(nose[0] - eye_mid[0])
        # Roll-correct: project the offset onto the perpendicular of the eye line
        roll = atan2(float(eye_vector[1]), float(eye_vector[0]))
        cos_r, sin_r = np.cos(-roll), np.sin(-roll)
        rotated_x = cos_r * nose_offset_x - sin_r * float(nose[1] - eye_mid[1])
        ratio = max(-1.0, min(1.0, rotated_x / (inter_eye * 0.5)))
        return float(degrees(atan2(ratio, 1.0)) * 2.0)

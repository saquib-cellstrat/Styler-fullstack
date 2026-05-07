import cv2
import numpy as np

from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage
from app.services.extraction.quality_gate import FrontalFaceGate
from app.shared.image_processing import ImageProcessor


class RetinaFaceAlignmentStage(AbstractPipelineStage):
    name = "alignment"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        self._settings = settings
        self._gate = FrontalFaceGate(models.retinaface, settings)
        self._image_processor = ImageProcessor()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if context.base_rgb is None:
            context.base_rgb = self._image_processor.decode(context.base_image_bytes)
        if context.donor_rgb is None:
            context.donor_rgb = self._image_processor.decode(context.donor_image_bytes)

        context.base_face = self._gate.evaluate(context.base_rgb)
        context.donor_face = self._gate.evaluate(context.donor_rgb)
        context.base_scalp_anchors = self._build_scalp_anchors(context.base_face.landmarks)
        context.donor_scalp_anchors = self._build_scalp_anchors(context.donor_face.landmarks)
        return context

    @staticmethod
    def _build_scalp_anchors(landmarks: np.ndarray) -> np.ndarray:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]
        eye_mid = (left_eye + right_eye) / 2.0
        eye_vec = right_eye - left_eye
        eye_dist = float(np.linalg.norm(eye_vec)) or 1.0
        up = np.array([0.0, -1.0], dtype=np.float32)
        roll = float(np.arctan2(eye_vec[1], eye_vec[0]))
        rot = np.array(
            [
                [np.cos(roll), -np.sin(roll)],
                [np.sin(roll), np.cos(roll)],
            ],
            dtype=np.float32,
        )
        local_up = rot @ up

        crown = eye_mid + local_up * (1.5 * eye_dist)
        left_temple = left_eye + local_up * (0.55 * eye_dist)
        right_temple = right_eye + local_up * (0.55 * eye_dist)
        fringe_center = eye_mid + local_up * (0.75 * eye_dist)
        left_side = mouth_left + np.array([-0.4 * eye_dist, -0.3 * eye_dist], dtype=np.float32)
        right_side = mouth_right + np.array([0.4 * eye_dist, -0.3 * eye_dist], dtype=np.float32)

        anchors = np.vstack(
            [left_temple, fringe_center, right_temple, crown, nose, left_side, right_side]
        ).astype(np.float32)
        return anchors

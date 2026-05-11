import numpy as np

from app.debug.exporter import DebugExporter
from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage
from app.services.cranial_hull import CranialHullEstimator
from app.services.extraction.quality_gate import FrontalFaceGate
from app.services.landmarks import DenseLandmarkService
from app.services.landmarks.debug_viz import render_landmarks_overlay
from app.shared.image_processing import ImageProcessor


class RetinaFaceAlignmentStage(AbstractPipelineStage):
    name = "alignment"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        self._settings = settings
        self._gate = FrontalFaceGate(models.retinaface, settings)
        self._image_processor = ImageProcessor()
        self._dense = DenseLandmarkService()
        self._hull = CranialHullEstimator()
        self._debug = DebugExporter(settings.debug_export_enabled, settings.debug_export_dir)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if context.base_rgb is None:
            context.base_rgb = self._image_processor.decode(context.base_image_bytes)
        if context.donor_rgb is None:
            context.donor_rgb = self._image_processor.decode(context.donor_image_bytes)

        context.base_face = self._gate.evaluate(context.base_rgb)
        context.donor_face = self._gate.evaluate(context.donor_rgb)
        canonical_base: np.ndarray | None = None
        canonical_donor: np.ndarray | None = None
        if self._settings.enable_experimental_geometry:
            context.base_dense_landmarks = self._dense.estimate(
                context.base_rgb, context.base_face.box, context.base_face.landmarks
            )
            context.donor_dense_landmarks = self._dense.estimate(
                context.donor_rgb, context.donor_face.box, context.donor_face.landmarks
            )
            context.base_cranial_hull = self._hull.estimate(
                context.base_rgb.shape[:2], context.base_dense_landmarks
            )
            context.donor_cranial_hull = self._hull.estimate(
                context.donor_rgb.shape[:2], context.donor_dense_landmarks
            )
            canonical_base = context.base_cranial_hull.canonical_anchors
            canonical_donor = context.donor_cranial_hull.canonical_anchors
        context.base_scalp_anchors = self._build_scalp_anchors(
            context.base_face.landmarks,
            context.base_face.box,
            context.base_rgb.shape[:2],
            canonical_base,
        )
        context.donor_scalp_anchors = self._build_scalp_anchors(
            context.donor_face.landmarks,
            context.donor_face.box,
            context.donor_rgb.shape[:2],
            canonical_donor,
        )
        if self._settings.enable_experimental_geometry and context.base_dense_landmarks is not None:
            context.debug_artifacts.paths["base_landmarks"] = self._debug.write_rgb(
                "base_landmarks",
                render_landmarks_overlay(context.base_rgb, context.base_dense_landmarks.points),
            )
            context.debug_artifacts.paths["donor_landmarks"] = self._debug.write_rgb(
                "donor_landmarks",
                render_landmarks_overlay(context.donor_rgb, context.donor_dense_landmarks.points),
            )
            context.debug_artifacts.paths["base_cranial_hull"] = self._debug.write_mask(
                "base_cranial_hull", context.base_cranial_hull.hull_mask
            )
            context.debug_artifacts.paths["donor_cranial_hull"] = self._debug.write_mask(
                "donor_cranial_hull", context.donor_cranial_hull.hull_mask
            )
        if self._settings.enable_experimental_geometry and self._settings.export_landmarks_json:
            self._dense.save_json(
                context.base_dense_landmarks, f"{self._settings.debug_export_dir}/base_landmarks.json"
            )
            self._dense.save_json(
                context.donor_dense_landmarks, f"{self._settings.debug_export_dir}/donor_landmarks.json"
            )
        return context

    @staticmethod
    def _build_scalp_anchors(
        landmarks: np.ndarray,
        face_box: tuple[float, float, float, float],
        image_shape: tuple[int, int],
        canonical_anchors: np.ndarray | None = None,
    ) -> np.ndarray:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]
        x1, y1, x2, y2 = face_box
        face_h = max(1.0, y2 - y1)
        face_w = max(1.0, x2 - x1)
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

        # Skull-aware anchor placement: combine eye distance + detected face box.
        scalp_lift = max(0.52 * eye_dist, 0.28 * face_h)
        crown = eye_mid + local_up * scalp_lift
        temple_lift = max(0.20 * eye_dist, 0.10 * face_h)
        left_temple = np.array(
            [x1 + 0.16 * face_w, left_eye[1]], dtype=np.float32
        ) + local_up * temple_lift
        right_temple = np.array(
            [x2 - 0.16 * face_w, right_eye[1]], dtype=np.float32
        ) + local_up * temple_lift
        fringe_center = eye_mid + local_up * max(0.24 * eye_dist, 0.12 * face_h)
        # Keep lower anchors soft and landmark-driven; hard jaw/chin anchors
        # can over-constrain TPS and cause visible cheek artifacts.
        mouth_center = (mouth_left + mouth_right) / 2.0
        left_side = np.array(
            [x1 + 0.12 * face_w, y1 + 0.70 * face_h], dtype=np.float32
        )
        right_side = np.array(
            [x2 - 0.12 * face_w, y1 + 0.70 * face_h], dtype=np.float32
        )
        left_cheek = np.array(
            [x1 + 0.20 * face_w, y1 + 0.58 * face_h], dtype=np.float32
        )
        right_cheek = np.array(
            [x2 - 0.20 * face_w, y1 + 0.58 * face_h], dtype=np.float32
        )
        lower_center = np.array([mouth_center[0], y1 + face_h * 0.88], dtype=np.float32)

        anchors = np.vstack(
            [
                left_temple,
                fringe_center,
                right_temple,
                crown,
                nose,
                left_side,
                right_side,
                left_cheek,
                right_cheek,
                lower_center,
            ]
        ).astype(np.float32)
        if canonical_anchors is not None and canonical_anchors.size > 0:
            extra = canonical_anchors.astype(np.float32)
            anchors = np.vstack([anchors, extra])
        h, w = image_shape
        anchors[:, 0] = np.clip(anchors[:, 0], 0.0, float(w - 1))
        anchors[:, 1] = np.clip(anchors[:, 1], 0.0, float(h - 1))
        return anchors

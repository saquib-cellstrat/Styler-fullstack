from pathlib import Path

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
from app.services.landmarks.head_anchors import build_head_anchors
from app.services.landmarks.mesh_landmarker import MeshLandmarker, MeshResult
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
        self._mesh: MeshLandmarker | None = None
        if settings.dense_landmark_backend == "mediapipe":
            task_path = Path(settings.model_weights_dir) / settings.face_landmarker_task_filename
            if task_path.exists():
                self._mesh = MeshLandmarker.get(str(task_path))

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if context.base_rgb is None:
            context.base_rgb = self._image_processor.decode(context.base_image_bytes)
        if context.donor_rgb is None:
            context.donor_rgb = self._image_processor.decode(context.donor_image_bytes)

        context.base_face = self._gate.evaluate(context.base_rgb)
        context.donor_face = self._gate.evaluate(context.donor_rgb)

        # Prefer the dense mesh for anchors when it resolves on BOTH images,
        # so the source/target anchor sets correspond index-for-index. If
        # either side has no mesh, both fall back to the legacy estimator.
        if self._mesh is not None:
            base_mesh = self._mesh.detect(context.base_rgb)
            donor_mesh = self._mesh.detect(context.donor_rgb)
            if base_mesh is not None and donor_mesh is not None:
                self._apply_mesh_anchors(context, base_mesh, donor_mesh)
                return context

        self._apply_legacy_anchors(context)
        return context

    def _apply_mesh_anchors(
        self, context: ProcessingContext, base_mesh: MeshResult, donor_mesh: MeshResult
    ) -> None:
        context.base_mesh_points = base_mesh.points
        context.donor_mesh_points = donor_mesh.points
        context.base_scalp_anchors = build_head_anchors(
            base_mesh.points, context.base_rgb.shape[:2]
        )
        context.donor_scalp_anchors = build_head_anchors(
            donor_mesh.points, context.donor_rgb.shape[:2]
        )
        if self._settings.debug_export_enabled:
            context.debug_artifacts.paths["base_landmarks"] = self._debug.write_rgb(
                "base_landmarks",
                render_landmarks_overlay(context.base_rgb, base_mesh.points),
            )
            context.debug_artifacts.paths["donor_landmarks"] = self._debug.write_rgb(
                "donor_landmarks",
                render_landmarks_overlay(context.donor_rgb, donor_mesh.points),
            )

    def _apply_legacy_anchors(self, context: ProcessingContext) -> None:
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

        # Skull-aware anchor placement: yaw-adaptive landmark-driven.
        # This uses the bounding box (x1, x2) to determine the true left/right
        # extent of the face, which perfectly adapts to yaw (turned faces) 
        # while using face_vert to prevent vertical elongation.
        right_dir = eye_vec / eye_dist
        mouth_center = (mouth_left + mouth_right) / 2.0
        face_vert = float(np.linalg.norm(mouth_center - eye_mid)) or 1.0

        pad_left = max(0.0, float(left_eye[0] - x1))
        pad_right = max(0.0, float(x2 - right_eye[0]))
        # Fallback if bounding box is tight or face is highly cropped
        pad_left = max(pad_left, 0.5 * eye_dist)
        pad_right = max(pad_right, 0.5 * eye_dist)

        # Crown: prevents elongation by using stable vertical face distance
        crown = eye_mid + local_up * (1.6 * face_vert)
        fringe_center = eye_mid + local_up * (0.6 * face_vert)

        # Temples
        left_temple = left_eye - right_dir * (pad_left * 0.85) + local_up * (0.3 * face_vert)
        right_temple = right_eye + right_dir * (pad_right * 0.85) + local_up * (0.3 * face_vert)

        # Ears/Sides - sit on the bounding box edge to prevent inward squeezing
        left_side = left_eye - right_dir * (pad_left * 1.0) - local_up * (0.1 * face_vert)
        right_side = right_eye + right_dir * (pad_right * 1.0) - local_up * (0.1 * face_vert)

        # Cheeks - gently taper inward
        left_cheek = left_eye - right_dir * (pad_left * 0.9) - local_up * (0.6 * face_vert)
        right_cheek = right_eye + right_dir * (pad_right * 0.9) - local_up * (0.6 * face_vert)

        # Jaw - wider than before to ensure hair doesn't bleed onto the mouth/nose
        left_jaw = left_eye - right_dir * (pad_left * 0.8) - local_up * (1.1 * face_vert)
        right_jaw = right_eye + right_dir * (pad_right * 0.8) - local_up * (1.1 * face_vert)

        # Chin
        lower_center = mouth_center - local_up * (0.4 * face_vert)

        # Shoulders / Neck (Prevents long hair from being stripped/squashed at the bottom)
        left_shoulder = left_eye - right_dir * (pad_left * 1.4) - local_up * (2.5 * face_vert)
        right_shoulder = right_eye + right_dir * (pad_right * 1.4) - local_up * (2.5 * face_vert)
        bottom_center = mouth_center - local_up * (2.5 * face_vert)

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
                left_jaw,
                right_jaw,
                lower_center,
                left_shoulder,
                right_shoulder,
                bottom_center,
            ]
        ).astype(np.float32)
        if canonical_anchors is not None and canonical_anchors.size > 0:
            extra = canonical_anchors.astype(np.float32)
            anchors = np.vstack([anchors, extra])
        h, w = image_shape
        anchors[:, 0] = np.clip(anchors[:, 0], 0.0, float(w - 1))
        anchors[:, 1] = np.clip(anchors[:, 1], 0.0, float(h - 1))
        return anchors

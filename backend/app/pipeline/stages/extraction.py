import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.model_registry import InferenceRegistry, LoadedOnnxModel
from app.core.onnx_session import resolve_modnet_output_name
from app.core.settings import Settings
from app.pipeline.contracts import HairRegions
from app.pipeline.context import AlphaMatte, ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage
from app.services.extraction.hair_parser import HairParser
from app.shared.image_processing import ImageProcessor


class ModNetExtractionStage(AbstractPipelineStage):
    name = "extraction"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        self._models = models
        self._settings = settings
        self._image_processor = ImageProcessor()
        self._hair_parser = HairParser(models.face_parser, settings)
        self._modnet_output_name = (
            resolve_modnet_output_name(models.modnet, settings.modnet_output_name)
            if models.modnet is not None
            else ""
        )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if context.donor_rgb is None:
            context.donor_rgb = self._image_processor.decode(context.donor_image_bytes)
        if context.donor_face is None:
            raise ValueError("Alignment stage must run before extraction")

        donor_rgb = context.donor_rgb
        hair_alpha = self._hair_parser.predict_hair_alpha(
            donor_rgb, face_box=context.donor_face.box
        )
        if self._settings.refine_hair_with_portrait_matte and self._models.modnet is not None:
            hair_alpha = self._refine_with_modnet(donor_rgb, hair_alpha, self._models.modnet)
        if self._settings.hair_alpha_gain != 1.0:
            hair_alpha = np.clip(
                hair_alpha * self._settings.hair_alpha_gain, 0.0, 1.0
            ).astype(np.float32)
        hair_alpha = self._stabilize_hair_alpha(hair_alpha)
        # Remove low-alpha donor background colors to avoid white halo contamination downstream.
        donor_rgb_clean = donor_rgb.copy()
        donor_rgb_clean[hair_alpha < 0.14] = 0

        donor_hair_rgba = self._image_processor.compose_rgba(
            donor_rgb_clean,
            hair_alpha,
            zero_rgb_where_transparent=self._settings.zero_rgb_where_transparent,
        )
        context.donor_hair_alpha = hair_alpha
        context.donor_hair_rgba = donor_hair_rgba
        context.donor_hair_regions = self._decompose_regions(
            hair_alpha=hair_alpha,
            landmarks=context.donor_face.landmarks,
        )
        context.donor_regions_v2 = self._decompose_regions_v2(
            hair_alpha=hair_alpha,
            landmarks=context.donor_face.landmarks,
            face_box=context.donor_face.box,
        )
        return context

    def _decompose_regions_v2(
        self,
        hair_alpha: AlphaMatte,
        landmarks: NDArray[np.float32],
        face_box: tuple[float, float, float, float],
    ) -> HairRegions:
        h, w = hair_alpha.shape
        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        eye_mid_y = float((left_eye[1] + right_eye[1]) * 0.5)
        x1, y1, x2, y2 = face_box
        face_h = max(1.0, y2 - y1)
        face_w = max(1.0, x2 - x1)

        # Use face-relative coordinates to avoid region drift in off-center crops.
        face_x = np.clip((x_grid - float(x1)) / face_w, -0.5, 1.5)
        face_y = np.clip((y_grid - float(y1)) / face_h, -0.5, 2.0)
        left_x = float(min(left_eye[0], right_eye[0]))
        right_x = float(max(left_eye[0], right_eye[0]))
        side_gate = (x_grid < left_x) | (x_grid > right_x)

        roots = (((face_y <= 0.20) | (y_grid < eye_mid_y - 0.06 * face_h)) * hair_alpha).astype(
            np.float32
        )
        forehead_line = (
            ((face_y > 0.16) & (face_y <= 0.42) & (y_grid < eye_mid_y + 0.05 * face_h)) * hair_alpha
        ).astype(np.float32)
        temples = (
            (side_gate & (face_y >= 0.10) & (face_y <= 0.52) & ((face_x < 0.26) | (face_x > 0.74)))
            * hair_alpha
        ).astype(np.float32)
        side_strands = (
            (side_gate & (face_y > 0.46) & (face_y <= 1.10) & ((face_x < 0.28) | (face_x > 0.72)))
            * hair_alpha
        ).astype(np.float32)
        long_strands = ((face_y > 0.88) * hair_alpha).astype(np.float32)
        shoulder_overlap = ((face_y > 1.08) * hair_alpha).astype(np.float32)
        return HairRegions(
            roots=np.clip(roots, 0.0, 1.0),
            forehead_line=np.clip(forehead_line, 0.0, 1.0),
            temples=np.clip(temples, 0.0, 1.0),
            side_strands=np.clip(side_strands, 0.0, 1.0),
            long_strands=np.clip(long_strands, 0.0, 1.0),
            shoulder_overlap=np.clip(shoulder_overlap, 0.0, 1.0),
        )

    def _decompose_regions(
        self,
        hair_alpha: AlphaMatte,
        landmarks: NDArray[np.float32],
    ) -> dict[str, AlphaMatte]:
        h, w = hair_alpha.shape
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        eye_mid_y = float((left_eye[1] + right_eye[1]) / 2.0)
        left_x = float(min(left_eye[0], right_eye[0]))
        right_x = float(max(left_eye[0], right_eye[0]))

        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        crown_mask = ((y_grid < eye_mid_y) * hair_alpha).astype(np.float32)
        fringe_mask = (
            ((y_grid >= eye_mid_y) & (y_grid <= eye_mid_y + h * 0.18)) * hair_alpha
        ).astype(np.float32)
        side_left = ((x_grid < left_x) * hair_alpha).astype(np.float32)
        side_right = ((x_grid > right_x) * hair_alpha).astype(np.float32)
        sides_mask = np.clip(side_left + side_right, 0.0, 1.0).astype(np.float32)
        return {
            "crown": crown_mask,
            "fringe": fringe_mask,
            "sides": sides_mask,
        }

    def _stabilize_hair_alpha(self, alpha: AlphaMatte) -> AlphaMatte:
        kernel_size = max(3, int(self._settings.hair_alpha_close_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        closed = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
        closed_f = closed.astype(np.float32) / 255.0
        stabilized = np.maximum(alpha, closed_f)

        core_mask = cv2.erode((stabilized > 0.35).astype(np.uint8), kernel, iterations=1) > 0
        core_floor = float(self._settings.hair_alpha_core_min_opacity)
        stabilized = np.where(core_mask, np.maximum(stabilized, core_floor), stabilized)
        interior_mask = cv2.erode((closed_f > 0.30).astype(np.uint8), kernel, iterations=1) > 0
        interior_floor = max(0.56, core_floor * 0.85)
        stabilized = np.where(
            interior_mask,
            np.maximum(stabilized, interior_floor),
            stabilized,
        )

        gamma = max(0.5, float(self._settings.hair_alpha_edge_gamma))
        gamma_adjusted = np.power(np.clip(stabilized, 0.0, 1.0), gamma).astype(np.float32)
        # Keep matte edge shaping from gamma, but do not dim the opaque interior.
        stabilized = np.where(core_mask, stabilized, gamma_adjusted).astype(np.float32)
        stabilized = np.where(stabilized > 0.96, 1.0, stabilized).astype(np.float32)
        return np.clip(stabilized, 0.0, 1.0).astype(np.float32)

    def _refine_with_modnet(
        self,
        rgb: NDArray[np.uint8],
        hair_alpha: AlphaMatte,
        modnet: LoadedOnnxModel,
    ) -> AlphaMatte:
        target_size = (
            self._settings.modnet_input_height,
            self._settings.modnet_input_width,
        )
        letterbox = self._image_processor.letterbox_for_modnet(
            rgb,
            target_size=target_size,
            mean=self._settings.modnet_mean_rgb,
            std=self._settings.modnet_std_rgb,
        )
        nchw = self._image_processor.to_nchw(letterbox.image)
        input_name = (
            self._settings.modnet_input_name
            if self._settings.modnet_input_name in modnet.input_names
            else modnet.input_names[0]
        )
        outputs = modnet.session.run([self._modnet_output_name], {input_name: nchw})
        portrait_alpha = np.asarray(outputs[0], dtype=np.float32)
        while portrait_alpha.ndim > 2:
            portrait_alpha = np.squeeze(portrait_alpha, axis=0)
        portrait_alpha = self._image_processor.remap_alpha_to_original(
            np.clip(portrait_alpha, 0.0, 1.0).astype(np.float32),
            letterbox,
        )

        low = self._settings.portrait_gate_low
        high = max(self._settings.portrait_gate_high, low + 1e-3)
        gate = np.clip((portrait_alpha - low) / (high - low), 0.0, 1.0)
        return np.clip(hair_alpha * gate, 0.0, 1.0).astype(np.float32)

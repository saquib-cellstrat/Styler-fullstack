"""Hair extraction stage.

Pipeline:
    1. Decode the donor image to RGB.
    2. Run a RetinaFace quality gate to ensure a frontal, sufficiently
       large face is present (no generative components).
    3. Run a CelebAMask-HQ BiSeNet face parser on a tight crop around the
       detected face to obtain a soft hair-only alpha matte (class 17,
       optionally combined with class 18 = hat).
    4. Optionally multiply that hair alpha by a MODNet portrait matte to
       suppress stray hair-class predictions outside the person silhouette.
    5. Compose RGBA, crop to the hair bounding box, and stream the PNG.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.model_registry import InferenceRegistry, LoadedOnnxModel
from app.core.onnx_session import resolve_modnet_output_name
from app.core.settings import Settings
from app.services.base import PipelineStage
from app.services.extraction.hair_parser import HairParser
from app.services.extraction.quality_gate import FaceDetection, FrontalFaceGate
from app.shared.image_processing import (
    AlphaMatte,
    ImageProcessor,
    RgbaImage,
)


@dataclass(frozen=True)
class ExtractionTimings:
    decode_ms: float
    detect_ms: float
    parse_ms: float
    refine_ms: float
    compose_ms: float
    total_ms: float


@dataclass(frozen=True)
class ExtractionResult:
    """Output artefact passed to downstream stages or serialized to PNG."""

    rgba: RgbaImage
    alpha: AlphaMatte
    face: FaceDetection
    timings: ExtractionTimings
    bbox: tuple[int, int, int, int] | None
    original_size: tuple[int, int]


class ExtractionService(PipelineStage[bytes, ExtractionResult]):
    """High-fidelity hair extraction stage."""

    name = "extraction"

    def __init__(self, registry: InferenceRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings
        self.image_processor = ImageProcessor()
        self.quality_gate = FrontalFaceGate(registry.retinaface, settings)
        self.hair_parser = HairParser(registry.face_parser, settings)
        self._modnet_output_name = (
            resolve_modnet_output_name(registry.modnet, settings.modnet_output_name)
            if registry.modnet is not None
            else ""
        )

    def process(self, payload: bytes) -> ExtractionResult:
        t_start = time.perf_counter()
        rgb = self.image_processor.decode(payload)
        t_decode = time.perf_counter()

        face = self.quality_gate.evaluate(rgb)
        t_detect = time.perf_counter()

        alpha = self.hair_parser.predict_hair_alpha(rgb, face_box=face.box)
        t_parse = time.perf_counter()

        if self.settings.refine_hair_with_portrait_matte and self.registry.modnet:
            alpha = self._refine_with_portrait_matte(rgb, alpha)
        gain = self.settings.hair_alpha_gain
        if gain != 1.0:
            alpha = np.clip(alpha * gain, 0.0, 1.0).astype(np.float32)
        alpha = self._stabilize_hair_alpha(alpha)
        t_refine = time.perf_counter()

        rgb_clean = rgb.copy()
        rgb_clean[alpha < 0.14] = 0

        rgba = self.image_processor.compose_rgba(
            rgb_clean,
            alpha,
            zero_rgb_where_transparent=self.settings.zero_rgb_where_transparent,
        )
        bbox: tuple[int, int, int, int] | None = None
        if self.settings.crop_output_to_hair_bbox:
            rgba, alpha, bbox = self.image_processor.crop_to_alpha_bbox(
                rgba,
                alpha,
                threshold=self.settings.alpha_visibility_threshold,
                padding_px=self.settings.output_bbox_padding_px,
            )
        t_compose = time.perf_counter()

        timings = ExtractionTimings(
            decode_ms=_ms(t_start, t_decode),
            detect_ms=_ms(t_decode, t_detect),
            parse_ms=_ms(t_detect, t_parse),
            refine_ms=_ms(t_parse, t_refine),
            compose_ms=_ms(t_refine, t_compose),
            total_ms=_ms(t_start, t_compose),
        )
        return ExtractionResult(
            rgba=rgba,
            alpha=alpha,
            face=face,
            timings=timings,
            bbox=bbox,
            original_size=(rgb.shape[0], rgb.shape[1]),
        )

    def encode_png(self, result: ExtractionResult) -> bytes:
        return self.image_processor.encode_png(
            result.rgba,
            compress_level=self.settings.png_compress_level,
            optimize=self.settings.png_optimize,
        )

    def _refine_with_portrait_matte(
        self, rgb: NDArray[np.uint8], hair_alpha: AlphaMatte
    ) -> AlphaMatte:
        target_size = (
            self.settings.modnet_input_height,
            self.settings.modnet_input_width,
        )
        letterbox = self.image_processor.letterbox_for_modnet(
            rgb,
            target_size=target_size,
            mean=self.settings.modnet_mean_rgb,
            std=self.settings.modnet_std_rgb,
        )
        nchw = self.image_processor.to_nchw(letterbox.image)
        assert self.registry.modnet is not None
        portrait_alpha_padded = self._run_modnet(self.registry.modnet, nchw)
        portrait_alpha = self.image_processor.remap_alpha_to_original(
            portrait_alpha_padded, letterbox
        )
        low = self.settings.portrait_gate_low
        high = max(self.settings.portrait_gate_high, low + 1e-3)
        gate = np.clip((portrait_alpha - low) / (high - low), 0.0, 1.0)
        return np.clip(hair_alpha * gate, 0.0, 1.0).astype(np.float32)

    def _run_modnet(
        self, model: LoadedOnnxModel, nchw: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        input_name = (
            self.settings.modnet_input_name
            if self.settings.modnet_input_name in model.input_names
            else model.input_names[0]
        )
        outputs = model.session.run(
            [self._modnet_output_name],
            {input_name: nchw},
        )
        alpha = np.asarray(outputs[0], dtype=np.float32)
        while alpha.ndim > 2:
            alpha = np.squeeze(alpha, axis=0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def _stabilize_hair_alpha(self, alpha: AlphaMatte) -> AlphaMatte:
        kernel_size = max(3, int(self.settings.hair_alpha_close_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        closed = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
        closed_f = closed.astype(np.float32) / 255.0
        stabilized = np.maximum(alpha, closed_f)

        core_mask = cv2.erode((stabilized > 0.35).astype(np.uint8), kernel, iterations=1) > 0
        core_floor = float(self.settings.hair_alpha_core_min_opacity)
        stabilized = np.where(core_mask, np.maximum(stabilized, core_floor), stabilized)
        interior_mask = cv2.erode((closed_f > 0.30).astype(np.uint8), kernel, iterations=1) > 0
        interior_floor = max(0.56, core_floor * 0.85)
        stabilized = np.where(
            interior_mask,
            np.maximum(stabilized, interior_floor),
            stabilized,
        )

        gamma = max(0.5, float(self.settings.hair_alpha_edge_gamma))
        gamma_adjusted = np.power(np.clip(stabilized, 0.0, 1.0), gamma).astype(np.float32)
        stabilized = np.where(core_mask, stabilized, gamma_adjusted).astype(np.float32)
        stabilized = np.where(stabilized > 0.96, 1.0, stabilized).astype(np.float32)
        return np.clip(stabilized, 0.0, 1.0).astype(np.float32)


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0

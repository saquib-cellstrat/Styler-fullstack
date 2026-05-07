"""Hair extraction stage.

Pipeline:
    1. Decode the donor image to RGB.
    2. Run a RetinaFace quality gate to ensure a frontal, sufficiently
       large face is present (no generative components).
    3. Run MODNet to obtain a portrait alpha matte and remap it back onto
       the original-resolution image to preserve hair-strand fidelity.
    4. Compose RGBA where the alpha channel carries the matte.

NOTE: MODNet is a portrait matting model - the alpha layer represents the
foreground person rather than only the hair. This module exposes the matte
as the RGBA alpha channel as specified by the v1 API; isolating "hair only"
without generative models is left to a future stage.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.model_registry import InferenceRegistry, LoadedOnnxModel
from app.core.onnx_session import resolve_modnet_output_name
from app.core.settings import Settings
from app.services.base import PipelineStage
from app.services.extraction.quality_gate import FaceDetection, FrontalFaceGate
from app.shared.image_processing import (
    AlphaMatte,
    ImageProcessor,
    RgbaImage,
)


@dataclass(frozen=True)
class ExtractionResult:
    """Output artefact passed to downstream stages or serialized to PNG."""

    rgba: RgbaImage
    alpha: AlphaMatte
    face: FaceDetection


class ExtractionService(PipelineStage[bytes, ExtractionResult]):
    """High-fidelity hair extraction stage.

    The ``process`` entry point conforms to ``PipelineStage`` so a future
    ``MasterPipeline`` can chain extraction -> warping -> blending without
    knowing anything about FastAPI.
    """

    name = "extraction"

    def __init__(self, registry: InferenceRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings
        self.image_processor = ImageProcessor()
        self.quality_gate = FrontalFaceGate(registry.retinaface, settings)
        self._modnet_output_name = resolve_modnet_output_name(
            registry.modnet,
            settings.modnet_output_name,
        )

    def process(self, payload: bytes) -> ExtractionResult:
        rgb = self.image_processor.decode(payload)
        face = self.quality_gate.evaluate(rgb)
        alpha = self._infer_alpha(rgb)
        rgba = self.image_processor.compose_rgba(rgb, alpha)
        return ExtractionResult(rgba=rgba, alpha=alpha, face=face)

    def encode_png(self, result: ExtractionResult) -> bytes:
        return self.image_processor.encode_png(result.rgba)

    def _infer_alpha(self, rgb: NDArray[np.uint8]) -> AlphaMatte:
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
        alpha_padded = self._run_modnet(self.registry.modnet, nchw)
        return self.image_processor.remap_alpha_to_original(alpha_padded, letterbox)

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
        # Flatten to (H, W); MODNet exports vary between (1,1,H,W) and (1,H,W).
        while alpha.ndim > 2:
            alpha = np.squeeze(alpha, axis=0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

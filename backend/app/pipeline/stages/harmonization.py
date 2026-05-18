import cv2
import numpy as np

from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage


class LabColorTransferStage(AbstractPipelineStage):
    name = "harmonization"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        _ = models
        self._settings = settings

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if (
            context.base_rgb is None
            or context.warped_hair_rgb is None
            or context.warped_hair_alpha is None
        ):
            raise ValueError("Warp stage must run before harmonization")

        # By default, pass the exact color from the donor image through
        # without any LAB statistical transfer, preserving 100% of the
        # original hair lighting, hue, and saturation exactly.
        context.harmonized_hair_rgb = context.warped_hair_rgb
        return context

    @staticmethod
    def _build_reference_mask(hair_mask: np.ndarray) -> np.ndarray:
        hair_u8 = (hair_mask.astype(np.uint8) * 255).astype(np.uint8)
        ring = cv2.dilate(hair_u8, np.ones((21, 21), dtype=np.uint8), iterations=1)
        ring = np.clip(ring - hair_u8, 0, 255)
        ref_mask = ring > 0
        if np.any(ref_mask):
            return ref_mask
        return ~hair_mask

    @staticmethod
    def _masked_stats(image_lab: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        stats: list[tuple[float, float]] = []
        for idx in range(3):
            values = image_lab[..., idx][mask]
            if values.size == 0:
                stats.append((0.0, 1.0))
            else:
                stats.append((float(np.mean(values)), float(np.std(values) + 1e-6)))
        return stats

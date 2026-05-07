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

        mask = context.warped_hair_alpha > 0.05
        if not np.any(mask):
            context.harmonized_hair_rgb = context.warped_hair_rgb
            return context

        source_lab = cv2.cvtColor(context.warped_hair_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        base_lab = cv2.cvtColor(context.base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        ref_mask = self._build_reference_mask(mask)
        source_stats = self._masked_stats(source_lab, mask)
        ref_stats = self._masked_stats(base_lab, ref_mask)

        transformed = source_lab.copy()
        for channel_idx in range(3):
            src_mean, src_std = source_stats[channel_idx]
            ref_mean, ref_std = ref_stats[channel_idx]
            channel = transformed[..., channel_idx]
            adjusted = ((channel - src_mean) * (ref_std / max(src_std, 1e-6))) + ref_mean
            transformed[..., channel_idx] = np.where(mask, adjusted, channel)

        harmonized = cv2.cvtColor(np.clip(transformed, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB)
        context.harmonized_hair_rgb = harmonized
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

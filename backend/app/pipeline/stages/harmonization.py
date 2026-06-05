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

        context.harmonized_hair_rgb = self._harmonize_lab(
            base_rgb=context.base_rgb,
            hair_rgb=context.warped_hair_rgb,
            hair_alpha=context.warped_hair_alpha,
        )
        return context

    def _harmonize_lab(
        self,
        base_rgb: np.ndarray,
        hair_rgb: np.ndarray,
        hair_alpha: np.ndarray,
    ) -> np.ndarray:
        alpha = np.clip(hair_alpha, 0.0, 1.0).astype(np.float32)
        alpha_threshold = float(getattr(self._settings, "harmonization_alpha_threshold", 0.12))
        min_pixels = int(getattr(self._settings, "harmonization_min_pixels", 64))
        hair_mask = alpha > alpha_threshold
        if int(np.count_nonzero(hair_mask)) < min_pixels:
            return hair_rgb

        reference_mask = self._build_reference_mask(alpha)
        if int(np.count_nonzero(reference_mask)) < min_pixels:
            return hair_rgb

        base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hair_lab = cv2.cvtColor(hair_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        base_stats = self._masked_stats(base_lab, reference_mask)
        hair_stats = self._masked_stats(hair_lab, hair_mask)

        luma_strength = float(getattr(self._settings, "harmonization_luma_strength", 0.42))
        contrast_strength = float(
            getattr(self._settings, "harmonization_contrast_strength", 0.22)
        )
        chroma_strength = float(getattr(self._settings, "harmonization_chroma_strength", 0.08))
        max_luma_shift = float(getattr(self._settings, "harmonization_max_luma_shift", 22.0))
        max_chroma_shift = float(
            getattr(self._settings, "harmonization_max_chroma_shift", 7.0)
        )

        adjusted = hair_lab.copy()
        adjusted[..., 0] = self._transfer_channel(
            hair_lab[..., 0],
            source_stats=hair_stats[0],
            reference_stats=base_stats[0],
            mean_strength=luma_strength,
            contrast_strength=contrast_strength,
            max_mean_shift=max_luma_shift,
        )
        for channel in (1, 2):
            adjusted[..., channel] = self._transfer_channel(
                hair_lab[..., channel],
                source_stats=hair_stats[channel],
                reference_stats=base_stats[channel],
                mean_strength=chroma_strength,
                contrast_strength=0.0,
                max_mean_shift=max_chroma_shift,
            )

        feather = np.clip((alpha - 0.02) / max(alpha_threshold - 0.02, 1e-3), 0.0, 1.0)
        adjusted = hair_lab * (1.0 - feather[..., None]) + adjusted * feather[..., None]
        adjusted_rgb = cv2.cvtColor(
            np.clip(adjusted, 0.0, 255.0).astype(np.uint8),
            cv2.COLOR_LAB2RGB,
        )
        return adjusted_rgb.astype(np.uint8)

    def _build_reference_mask(self, hair_alpha: np.ndarray) -> np.ndarray:
        threshold = float(getattr(self._settings, "harmonization_alpha_threshold", 0.12))
        kernel_size = max(3, int(getattr(self._settings, "harmonization_ring_kernel", 31)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        hair_u8 = (hair_alpha > threshold).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(hair_u8, kernel, iterations=1)
        ring = np.clip(dilated - hair_u8, 0, 1)
        ref_mask = ring > 0
        if np.any(ref_mask):
            return ref_mask
        return hair_u8 == 0

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

    @staticmethod
    def _transfer_channel(
        channel: np.ndarray,
        source_stats: tuple[float, float],
        reference_stats: tuple[float, float],
        mean_strength: float,
        contrast_strength: float,
        max_mean_shift: float,
    ) -> np.ndarray:
        source_mean, source_std = source_stats
        reference_mean, reference_std = reference_stats
        mean_shift = np.clip(reference_mean - source_mean, -max_mean_shift, max_mean_shift)
        target_mean = source_mean + mean_shift * np.clip(mean_strength, 0.0, 1.0)

        std_floor = max(source_std * 0.55, 1e-3)
        std_ceiling = max(source_std * 1.65, std_floor + 1e-3)
        clipped_reference_std = np.clip(reference_std, std_floor, std_ceiling)
        target_std = source_std + (clipped_reference_std - source_std) * np.clip(
            contrast_strength, 0.0, 1.0
        )
        scale = target_std / max(source_std, 1e-3)
        return (channel - source_mean) * scale + target_mean

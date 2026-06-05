import cv2
import numpy as np

from app.debug.exporter import DebugExporter
from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage
from app.services.compositing import AnatomicalCompositor
from app.services.depth import OptionalDepthEstimator


class LaplacianBlendStage(AbstractPipelineStage):
    name = "blending"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        _ = models
        self._settings = settings
        self._compositor = AnatomicalCompositor()
        self._depth = OptionalDepthEstimator()
        self._debug = DebugExporter(settings.debug_export_enabled, settings.debug_export_dir)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if (
            context.base_rgb is None
            or context.warped_hair_alpha is None
            or (context.harmonized_hair_rgb is None and context.warped_hair_rgb is None)
        ):
            raise ValueError("Harmonization stage must run before blending")

        base = context.base_rgb.astype(np.float32) / 255.0
        hair_rgb = (
            context.harmonized_hair_rgb
            if context.harmonized_hair_rgb is not None
            else context.warped_hair_rgb
        )
        assert hair_rgb is not None
        alpha = np.clip(context.warped_hair_alpha, 0.0, 1.0).astype(np.float32)
        hair_rgb = self._solidify_edge_rgb(hair_rgb, alpha)
        hair = hair_rgb.astype(np.float32) / 255.0
        alpha = self._prepare_alpha(alpha)
        depth_map = self._depth.estimate(context.base_rgb) if self._settings.enable_depth_occlusion else None
        if self._settings.enable_depth_occlusion and context.donor_regions_v2 is not None:
            context.occlusion_metadata = self._compositor.build_occlusion(
                context.base_rgb, alpha, context.donor_regions_v2, depth_map
            )
            alpha = self._compositor.apply_occlusion_to_alpha(alpha, context.occlusion_metadata)
            if context.occlusion_metadata.depth_map is not None:
                context.debug_artifacts.paths["depth_map"] = self._debug.write_mask(
                    "depth_map", context.occlusion_metadata.depth_map
                )
            context.debug_artifacts.paths["occlusion_front"] = self._debug.write_mask(
                "occlusion_front", context.occlusion_metadata.front_hair_mask
            )
            context.debug_artifacts.paths["occlusion_back"] = self._debug.write_mask(
                "occlusion_back", context.occlusion_metadata.back_hair_mask
            )

        shadow_alpha = self._contact_shadow(alpha)
        context.contact_shadow_alpha = shadow_alpha
        base_shadowed = np.clip(base * (1.0 - shadow_alpha[..., None]), 0.0, 1.0)

        blend = self._laplacian_blend(base_shadowed, hair, alpha)
        rgba = np.dstack(
            [
                np.clip(blend * 255.0, 0.0, 255.0).astype(np.uint8),
                np.full(alpha.shape, 255, dtype=np.uint8),
            ]
        )
        context.output_rgba = rgba
        return context

    def _laplacian_blend(self, base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        levels = max(2, self._settings.laplacian_pyramid_levels)
        gp_mask = [alpha]
        gp_base = [base]
        gp_overlay = [overlay]

        for _ in range(1, levels):
            gp_mask.append(cv2.pyrDown(gp_mask[-1]))
            gp_base.append(cv2.pyrDown(gp_base[-1]))
            gp_overlay.append(cv2.pyrDown(gp_overlay[-1]))

        lp_base = [gp_base[-1]]
        lp_overlay = [gp_overlay[-1]]
        for idx in range(levels - 1, 0, -1):
            size = (gp_base[idx - 1].shape[1], gp_base[idx - 1].shape[0])
            lp_base.append(gp_base[idx - 1] - cv2.pyrUp(gp_base[idx], dstsize=size))
            lp_overlay.append(gp_overlay[idx - 1] - cv2.pyrUp(gp_overlay[idx], dstsize=size))

        blended_levels: list[np.ndarray] = []
        for idx in range(levels):
            mask = gp_mask[levels - 1 - idx][..., None]
            blended = (lp_overlay[idx] * mask) + (lp_base[idx] * (1.0 - mask))
            blended_levels.append(blended)

        output = blended_levels[0]
        for idx in range(1, levels):
            size = (blended_levels[idx].shape[1], blended_levels[idx].shape[0])
            output = cv2.pyrUp(output, dstsize=size) + blended_levels[idx]
        return np.clip(output, 0.0, 1.0)

    @staticmethod
    def _prepare_alpha(alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        alpha_u8 = cv2.medianBlur(alpha_u8, 5)
        softened = cv2.GaussianBlur(alpha_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=1.0, sigmaY=1.0)
        softened = np.maximum(softened, alpha * 0.85)
        softened = np.where(softened > 0.55, np.maximum(softened, 0.82), softened)
        return np.clip(softened, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _solidify_edge_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        core = alpha > 0.18
        if not np.any(core):
            return rgb

        core_u8 = core.astype(np.uint8)
        band = cv2.dilate(core_u8, np.ones((17, 17), dtype=np.uint8), iterations=1) > 0
        fill_mask = np.logical_and(band, alpha <= 0.18)
        if not np.any(fill_mask):
            return rgb

        mask_u8 = (fill_mask.astype(np.uint8) * 255).astype(np.uint8)
        return cv2.inpaint(rgb, mask_u8, 3.0, cv2.INPAINT_TELEA).astype(np.uint8)

    def _contact_shadow(self, alpha: np.ndarray) -> np.ndarray:
        kernel = np.ones((9, 9), dtype=np.uint8)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        boundary = cv2.morphologyEx(alpha_u8, cv2.MORPH_GRADIENT, kernel)
        shifted = np.roll(boundary, shift=4, axis=0)
        shifted[:4, :] = 0
        blurred = cv2.GaussianBlur(
            shifted.astype(np.float32) / 255.0,
            ksize=(0, 0),
            sigmaX=self._settings.contact_shadow_blur_sigma,
            sigmaY=self._settings.contact_shadow_blur_sigma,
        )
        outside_bias = np.clip(1.0 - alpha * 0.75, 0.25, 1.0)
        return np.clip(blurred * outside_bias * self._settings.contact_shadow_opacity, 0.0, 1.0)

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
        hair = hair_rgb.astype(np.float32) / 255.0
        if self._settings.enable_edge_decontamination:
            hair = self._decontaminate_edges(hair, context.warped_hair_alpha)
        alpha = np.clip(context.warped_hair_alpha, 0.0, 1.0).astype(np.float32)
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

    def _decontaminate_edges(self, hair: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Bleed confident hair colour into the soft edge band.

        Feathered edge pixels carry a mix of true hair and the donor's
        background, which reads as a grey halo once composited. Replacing the
        band's colour with the nearest confident-hair colour makes the soft
        edge fade in hair tone instead, while the alpha still does the
        feathering.
        """
        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        core = (a > 0.7).astype(np.uint8)
        if int(core.sum()) == 0:
            return hair
        band = max(1, int(self._settings.edge_decontam_band_px))
        reach = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=band)
        target = ((reach > 0) & (core == 0))
        if not np.any(target):
            return hair

        filled = hair.copy()
        known = core.astype(np.float32)
        k = (3, 3)
        for _ in range(band):
            num = cv2.blur(filled * known[..., None], k)
            den = cv2.blur(known, k)[..., None] + 1e-6
            avg = num / den
            grown = (cv2.dilate(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            newly = grown & (known < 0.5) & target
            filled = np.where(newly[..., None], avg, filled)
            known = np.where(newly, 1.0, known).astype(np.float32)
        return np.clip(filled, 0.0, 1.0).astype(np.float32)

    def _prepare_alpha(self, alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        alpha_u8 = cv2.medianBlur(alpha_u8, 5)
        softened = cv2.GaussianBlur(alpha_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=1.0, sigmaY=1.0)
        softened = np.where(softened > 0.55, np.maximum(softened, 0.82), softened)
        # Floor the faint tail to remove the halo/background haze around hair.
        low = self._settings.alpha_floor_low
        high = max(self._settings.alpha_floor_high, low + 1e-3)
        ramp = np.clip((softened - low) / (high - low), 0.0, 1.0)
        softened = softened * ramp
        softened = self._crop_halo(softened)
        return np.clip(softened, 0.0, 1.0).astype(np.float32)

    def _crop_halo(self, alpha: np.ndarray) -> np.ndarray:
        """Limit alpha to a band around the solid hair, killing the far haze.

        A backlit donor matte spreads faint alpha well beyond the hair, which
        shows as a wide grey cloud over the background. Real strands hug the
        silhouette, so attenuating alpha by distance from the solid core
        removes the cloud while keeping edge wisps intact. Short, clean styles
        have no far haze and are unaffected.
        """
        core_dist = float(self._settings.halo_crop_core_dist_px)
        falloff = max(1.0, float(self._settings.halo_crop_falloff_px))
        solid = (alpha > 0.5).astype(np.uint8)
        if int(solid.sum()) == 0:
            return alpha
        dist = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 3)
        atten = np.clip(1.0 - (dist - core_dist) / falloff, 0.0, 1.0).astype(np.float32)
        return alpha * atten

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
        return np.clip(blurred * self._settings.contact_shadow_opacity, 0.0, 1.0)

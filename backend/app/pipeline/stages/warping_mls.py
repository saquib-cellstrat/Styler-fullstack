import cv2
import numpy as np

from app.debug.exporter import DebugExporter
from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage
from app.services.warping.mls_rigid import build_backward_rigid_mls_remap_maps
from app.services.warping.region_tps import RegionAwareTpsWarper


class MlsWarpStage(AbstractPipelineStage):
    """Warp donor hair onto the base using rigid Moving Least Squares (inverse MLS maps)."""

    name = "warp"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        _ = models
        self._settings = settings
        self._region = RegionAwareTpsWarper()
        self._debug = DebugExporter(settings.debug_export_enabled, settings.debug_export_dir)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if (
            context.base_rgb is None
            or context.donor_hair_rgba is None
            or context.donor_hair_alpha is None
            or context.base_scalp_anchors is None
            or context.donor_scalp_anchors is None
        ):
            raise ValueError("Alignment and extraction stages must run before warping")

        donor_rgb = context.donor_hair_rgba[..., :3]
        donor_alpha = context.donor_hair_alpha
        base_h, base_w = context.base_rgb.shape[:2]

        target_pts = context.base_scalp_anchors.astype(np.float32)
        if self._settings.enable_region_aware_tps and context.donor_regions_v2 is not None:
            blended, stiffness_map = self._region.weighted_anchor_targets(
                context.donor_scalp_anchors.astype(np.float32),
                context.base_scalp_anchors.astype(np.float32),
                context.donor_regions_v2,
            )[:2]
            target_pts = blended
            context.debug_artifacts.paths["warp_stiffness"] = self._debug.write_mask(
                "warp_stiffness", stiffness_map
            )

        map_x, map_y = build_backward_rigid_mls_remap_maps(
            context.donor_scalp_anchors.astype(np.float32),
            target_pts,
            (base_h, base_w),
            donor_rgb.shape[:2],
            grid_long_edge=self._settings.mls_map_grid_long_edge,
            max_iterations=self._settings.mls_inverse_max_iterations,
            alpha=self._settings.mls_weight_alpha,
            eps=self._settings.mls_weight_eps,
        )
        context.warp_plan = None

        warped_rgb = cv2.remap(
            donor_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_alpha = cv2.remap(
            donor_alpha,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_alpha = self._stabilize_warped_alpha(warped_alpha)

        context.warped_hair_rgb = np.clip(warped_rgb, 0, 255).astype(np.uint8)
        context.warped_hair_alpha = np.clip(warped_alpha, 0.0, 1.0).astype(np.float32)
        return context

    @staticmethod
    def _stabilize_warped_alpha(alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        close_kernel = np.ones((5, 5), dtype=np.uint8)
        closed = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, close_kernel)
        feather = cv2.GaussianBlur(closed.astype(np.float32) / 255.0, (0, 0), sigmaX=1.2, sigmaY=1.2)
        core = cv2.erode((feather > 0.4).astype(np.uint8), close_kernel, iterations=1) > 0
        feather = np.where(core, np.maximum(feather, 0.78), feather)
        return np.clip(feather, 0.0, 1.0).astype(np.float32)

import cv2
import numpy as np
from numpy.typing import NDArray

from app.debug.exporter import DebugExporter
from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage
from app.services.warping.region_tps import RegionAwareTpsWarper


class TpsWarpStage(AbstractPipelineStage):
    name = "warp"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        _ = models
        self._settings = settings
        self._warper = RegionAwareTpsWarper()
        self._debug = DebugExporter(
            getattr(settings, "debug_export_enabled", False),
            getattr(settings, "debug_export_dir", "debug/exports"),
        )

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

        if (
            getattr(self._settings, "enable_region_aware_tps", False)
            and context.donor_regions_v2 is not None
        ):
            plan = self._warper.build_plan(
                source_points=context.donor_scalp_anchors,
                target_points=context.base_scalp_anchors,
                regions=context.donor_regions_v2,
                output_shape=(base_h, base_w),
                regularization=self._settings.tps_regularization,
            )
            context.warp_plan = plan
            warped_rgb, warped_alpha = self._warper.warp_rgba(donor_rgb, donor_alpha, plan)
            context.debug_artifacts.paths["warp_stiffness"] = self._debug.write_mask(
                "warp_stiffness", plan.stiffness_map
            )
        else:
            map_x, map_y = self._build_tps_maps(
                source_points=context.donor_scalp_anchors,
                target_points=context.base_scalp_anchors,
                output_shape=(base_h, base_w),
            )
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

    def _build_tps_maps(
        self,
        source_points: NDArray[np.float32],
        target_points: NDArray[np.float32],
        output_shape: tuple[int, int],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        h, w = output_shape
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        eval_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        fx = self._solve_tps(target_points, source_points[:, 0])
        fy = self._solve_tps(target_points, source_points[:, 1])

        map_x = self._evaluate_tps(target_points, eval_points, fx).reshape(h, w)
        map_y = self._evaluate_tps(target_points, eval_points, fy).reshape(h, w)
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def _solve_tps(
        self,
        control_points: NDArray[np.float32],
        values: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        n = control_points.shape[0]
        k = self._kernel(control_points, control_points)
        p = np.concatenate([np.ones((n, 1), dtype=np.float64), control_points], axis=1)
        top = np.concatenate(
            [k + np.eye(n, dtype=np.float64) * self._settings.tps_regularization, p],
            axis=1,
        )
        bottom = np.concatenate([p.T, np.zeros((3, 3), dtype=np.float64)], axis=1)
        lhs = np.concatenate([top, bottom], axis=0)
        rhs = np.concatenate([values.astype(np.float64), np.zeros(3, dtype=np.float64)])
        return np.linalg.solve(lhs, rhs)

    def _evaluate_tps(
        self,
        control_points: NDArray[np.float32],
        query_points: NDArray[np.float32],
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n = control_points.shape[0]
        weights = params[:n]
        affine = params[n:]
        k = self._kernel(query_points, control_points)
        p = np.concatenate(
            [np.ones((query_points.shape[0], 1), dtype=np.float64), query_points], axis=1
        )
        return k @ weights + p @ affine

    @staticmethod
    def _kernel(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float64]:
        diff = a[:, None, :] - b[None, :, :]
        r2 = np.sum(diff * diff, axis=2, dtype=np.float64)
        return r2 * np.log(r2 + 1e-6)

    def _stabilize_warped_alpha(self, alpha: NDArray[np.float32]) -> NDArray[np.float32]:
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        kernel_size = max(3, int(getattr(self._settings, "hair_alpha_close_kernel", 5)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, close_kernel)
        closed_f = closed.astype(np.float32) / 255.0
        feather = cv2.GaussianBlur(closed_f, (0, 0), sigmaX=1.2, sigmaY=1.2)
        core = cv2.erode((alpha > 0.5).astype(np.uint8), close_kernel, iterations=1) > 0
        core_floor = float(getattr(self._settings, "hair_alpha_core_min_opacity", 0.72))
        feather = np.where(core, np.maximum(feather, core_floor), feather)
        interior_mask = cv2.erode((closed_f > 0.32).astype(np.uint8), close_kernel, iterations=1) > 0
        interior_floor = max(0.56, core_floor * 0.85)
        feather = np.where(interior_mask, np.maximum(feather, interior_floor), feather)
        preserve_mask = alpha > 0.80
        feather = np.where(preserve_mask, np.maximum(feather, alpha), feather)
        feather = np.where(feather > 0.96, 1.0, feather)
        return np.clip(feather, 0.0, 1.0).astype(np.float32)

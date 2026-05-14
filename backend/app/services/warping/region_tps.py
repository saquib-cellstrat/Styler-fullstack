from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipeline.contracts import HairRegions, WarpPlan


class RegionAwareTpsWarper:
    def weighted_anchor_targets(
        self,
        source_points: NDArray[np.float32],
        target_points: NDArray[np.float32],
        regions: HairRegions,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str], NDArray[np.float32]]:
        """Blend each anchor toward donor (stiff) or base (soft) without building TPS."""
        region_ids = self._region_labels(source_points.shape[0])
        stiffness_lookup = {
            "roots": 0.95,
            "forehead_line": 0.9,
            "temples": 0.8,
            "side_strands": 0.55,
            "long_strands": 0.35,
            "shoulder_overlap": 0.5,
        }
        stiffness = np.asarray([stiffness_lookup.get(name, 0.6) for name in region_ids], dtype=np.float32)
        blended = source_points * stiffness[:, None] + target_points * (1.0 - stiffness[:, None])
        return blended.astype(np.float32), self._stiffness_map(regions), region_ids, stiffness

    def build_plan(
        self,
        source_points: NDArray[np.float32],
        target_points: NDArray[np.float32],
        regions: HairRegions,
        output_shape: tuple[int, int],
        regularization: float,
    ) -> WarpPlan:
        h, w = output_shape
        weighted_targets, stiffness_map, region_ids, stiffness = self.weighted_anchor_targets(
            source_points, target_points, regions
        )
        map_x, map_y = self._build_tps_maps(source_points, weighted_targets, output_shape, regularization)
        return WarpPlan(
            source_points=source_points.astype(np.float32),
            target_points=weighted_targets.astype(np.float32),
            region_ids=region_ids,
            stiffness=stiffness,
            stiffness_map=stiffness_map,
            map_x=map_x,
            map_y=map_y,
        )

    def warp_rgba(
        self, rgb: NDArray[np.uint8], alpha: NDArray[np.float32], plan: WarpPlan
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        warped_rgb = cv2.remap(rgb, plan.map_x, plan.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_alpha = cv2.remap(alpha, plan.map_x, plan.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return warped_rgb.astype(np.uint8), np.clip(warped_alpha, 0.0, 1.0).astype(np.float32)

    def _stiffness_map(self, regions: HairRegions) -> NDArray[np.float32]:
        stiff = (
            regions.roots * 0.95
            + regions.forehead_line * 0.9
            + regions.temples * 0.8
            + regions.side_strands * 0.55
            + regions.long_strands * 0.35
            + regions.shoulder_overlap * 0.5
        )
        return np.clip(stiff, 0.0, 1.0).astype(np.float32)

    def _region_labels(self, n: int) -> list[str]:
        labels = ["roots", "forehead_line", "temples", "temples", "side_strands", "side_strands", "long_strands", "long_strands", "shoulder_overlap", "roots"]
        if n <= len(labels):
            return labels[:n]
        return labels + ["long_strands"] * (n - len(labels))

    def _build_tps_maps(
        self,
        source_points: NDArray[np.float32],
        target_points: NDArray[np.float32],
        output_shape: tuple[int, int],
        reg: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        h, w = output_shape
        y, x = np.indices((h, w), dtype=np.float32)
        q = np.stack([x.ravel(), y.ravel()], axis=1)
        fx = self._solve_tps(target_points, source_points[:, 0], reg)
        fy = self._solve_tps(target_points, source_points[:, 1], reg)
        map_x = self._eval_tps(target_points, q, fx).reshape(h, w).astype(np.float32)
        map_y = self._eval_tps(target_points, q, fy).reshape(h, w).astype(np.float32)
        return map_x, map_y

    def _solve_tps(self, cp: NDArray[np.float32], val: NDArray[np.float32], reg: float) -> NDArray[np.float64]:
        n = cp.shape[0]
        k = self._kernel(cp, cp)
        p = np.concatenate([np.ones((n, 1), dtype=np.float64), cp], axis=1)
        lhs = np.block([[k + np.eye(n) * reg, p], [p.T, np.zeros((3, 3), dtype=np.float64)]])
        rhs = np.concatenate([val.astype(np.float64), np.zeros(3, dtype=np.float64)])
        return np.linalg.solve(lhs, rhs)

    def _eval_tps(self, cp: NDArray[np.float32], q: NDArray[np.float32], prm: NDArray[np.float64]) -> NDArray[np.float64]:
        n = cp.shape[0]
        w = prm[:n]
        a = prm[n:]
        return self._kernel(q, cp) @ w + np.c_[np.ones((q.shape[0], 1)), q] @ a

    def _kernel(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float64]:
        d = a[:, None, :] - b[None, :, :]
        r2 = np.sum(d * d, axis=2, dtype=np.float64)
        return r2 * np.log(r2 + 1e-6)

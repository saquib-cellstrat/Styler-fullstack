from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipeline.contracts import HairRegions, OcclusionMetadata


class AnatomicalCompositor:
    def build_occlusion(
        self,
        base_rgb: NDArray[np.uint8],
        hair_alpha: NDArray[np.float32],
        regions: HairRegions,
        depth_map: NDArray[np.float32] | None,
    ) -> OcclusionMetadata:
        h, w = hair_alpha.shape
        regions = self._align_regions_to_shape(regions, (h, w))
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        neck = ((yy > 0.62).astype(np.float32) * (1.0 - regions.forehead_line)).astype(np.float32)
        shoulder = ((yy > 0.72).astype(np.float32) * (regions.long_strands > 0.05)).astype(np.float32)
        if depth_map is None:
            front = np.clip(regions.roots + regions.forehead_line + regions.temples, 0.0, 1.0).astype(np.float32)
        else:
            front = (depth_map < np.quantile(depth_map, 0.45)).astype(np.float32) * hair_alpha
        back = np.clip(hair_alpha - front, 0.0, 1.0).astype(np.float32)
        _ = base_rgb, w
        return OcclusionMetadata(
            neck_mask=neck,
            shoulder_mask=shoulder,
            front_hair_mask=np.clip(front, 0.0, 1.0).astype(np.float32),
            back_hair_mask=back,
            depth_map=depth_map,
        )

    def _align_regions_to_shape(
        self, regions: HairRegions, shape: tuple[int, int]
    ) -> HairRegions:
        h, w = shape
        return HairRegions(
            roots=self._resize_mask(regions.roots, (w, h)),
            forehead_line=self._resize_mask(regions.forehead_line, (w, h)),
            temples=self._resize_mask(regions.temples, (w, h)),
            side_strands=self._resize_mask(regions.side_strands, (w, h)),
            long_strands=self._resize_mask(regions.long_strands, (w, h)),
            shoulder_overlap=self._resize_mask(regions.shoulder_overlap, (w, h)),
        )

    def _resize_mask(
        self, mask: NDArray[np.float32], size: tuple[int, int]
    ) -> NDArray[np.float32]:
        if mask.shape == (size[1], size[0]):
            return np.clip(mask, 0.0, 1.0).astype(np.float32)
        resized = cv2.resize(
            np.clip(mask, 0.0, 1.0).astype(np.float32),
            size,
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    def apply_occlusion_to_alpha(
        self, hair_alpha: NDArray[np.float32], occlusion: OcclusionMetadata
    ) -> NDArray[np.float32]:
        suppress = np.clip(0.7 * occlusion.neck_mask + 0.9 * occlusion.shoulder_mask, 0.0, 1.0)
        back = np.clip(occlusion.back_hair_mask * (1.0 - suppress), 0.0, 1.0)
        return np.clip(back + occlusion.front_hair_mask, 0.0, 1.0).astype(np.float32)

    def render_mesh_overlay(
        self, rgb: NDArray[np.uint8], points: NDArray[np.float32]
    ) -> NDArray[np.uint8]:
        out = rgb.copy()
        for idx in range(1, points.shape[0]):
            p0 = tuple(np.round(points[idx - 1]).astype(int))
            p1 = tuple(np.round(points[idx]).astype(int))
            cv2.line(out, p0, p1, (255, 200, 0), 1, lineType=cv2.LINE_AA)
        return out

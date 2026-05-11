from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PointSet = NDArray[np.float32]
Mask = NDArray[np.float32]
RgbImage = NDArray[np.uint8]


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class DenseLandmarks:
    points: PointSet
    normalized_points: PointSet
    forehead_contour: PointSet
    jaw_contour: PointSet
    temple_left: tuple[float, float]
    temple_right: tuple[float, float]
    ear_left: tuple[float, float] | None
    ear_right: tuple[float, float] | None
    pose: HeadPose
    metadata: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class CranialHull:
    hull_mask: Mask
    scalp_arc: PointSet
    skull_ellipse: PointSet
    crown_point: tuple[float, float]
    forehead_curvature: float
    canonical_anchors: PointSet
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HairRegions:
    roots: Mask
    forehead_line: Mask
    temples: Mask
    side_strands: Mask
    long_strands: Mask
    shoulder_overlap: Mask


@dataclass(frozen=True)
class WarpPlan:
    source_points: PointSet
    target_points: PointSet
    region_ids: list[str]
    stiffness: NDArray[np.float32]
    stiffness_map: Mask
    map_x: NDArray[np.float32]
    map_y: NDArray[np.float32]


@dataclass(frozen=True)
class OcclusionMetadata:
    neck_mask: Mask
    shoulder_mask: Mask
    front_hair_mask: Mask
    back_hair_mask: Mask
    depth_map: Mask | None = None


@dataclass
class DebugArtifacts:
    paths: dict[str, Path | None] = field(default_factory=dict)


from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.pipeline.contracts import (
    CranialHull,
    DebugArtifacts,
    DenseLandmarks,
    HairRegions,
    OcclusionMetadata,
    WarpPlan,
)
from app.services.extraction.quality_gate import FaceDetection

RgbImage = NDArray[np.uint8]
RgbaImage = NDArray[np.uint8]
AlphaMatte = NDArray[np.float32]
PointSet = NDArray[np.float32]
RegionMap = dict[str, AlphaMatte]
TimingMap = dict[str, float]
ImplementationMap = dict[str, str]


@dataclass
class ProcessingContext:
    base_image_bytes: bytes
    donor_image_bytes: bytes
    base_rgb: RgbImage | None = None
    donor_rgb: RgbImage | None = None
    base_face: FaceDetection | None = None
    donor_face: FaceDetection | None = None
    base_scalp_anchors: PointSet | None = None
    donor_scalp_anchors: PointSet | None = None
    donor_hair_alpha: AlphaMatte | None = None
    donor_hair_rgba: RgbaImage | None = None
    donor_hair_regions: RegionMap = field(default_factory=dict)
    donor_regions_v2: HairRegions | None = None
    warped_hair_rgb: RgbImage | None = None
    warped_hair_alpha: AlphaMatte | None = None
    warp_plan: WarpPlan | None = None
    occlusion_metadata: OcclusionMetadata | None = None
    base_dense_landmarks: DenseLandmarks | None = None
    donor_dense_landmarks: DenseLandmarks | None = None
    base_mesh_points: PointSet | None = None
    donor_mesh_points: PointSet | None = None
    base_cranial_hull: CranialHull | None = None
    donor_cranial_hull: CranialHull | None = None
    debug_artifacts: DebugArtifacts = field(default_factory=DebugArtifacts)
    harmonized_hair_rgb: RgbImage | None = None
    contact_shadow_alpha: AlphaMatte | None = None
    output_rgba: RgbaImage | None = None
    timings_ms: TimingMap = field(default_factory=dict)
    selected_implementations: ImplementationMap = field(default_factory=dict)

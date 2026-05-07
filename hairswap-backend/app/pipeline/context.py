from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

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
    warped_hair_rgb: RgbImage | None = None
    warped_hair_alpha: AlphaMatte | None = None
    harmonized_hair_rgb: RgbImage | None = None
    contact_shadow_alpha: AlphaMatte | None = None
    output_rgba: RgbaImage | None = None
    timings_ms: TimingMap = field(default_factory=dict)
    selected_implementations: ImplementationMap = field(default_factory=dict)

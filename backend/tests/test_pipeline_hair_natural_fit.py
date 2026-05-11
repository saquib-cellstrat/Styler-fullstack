import numpy as np

from app.pipeline.stages.alignment import RetinaFaceAlignmentStage
from app.pipeline.stages.blending import LaplacianBlendStage
from app.pipeline.stages.extraction import ModNetExtractionStage
from app.pipeline.stages.warping import TpsWarpStage


class _AlphaSettings:
    hair_alpha_close_kernel = 5
    hair_alpha_core_min_opacity = 0.72
    hair_alpha_edge_gamma = 0.9


def test_alignment_adds_sideburn_and_jaw_guides() -> None:
    landmarks = np.array(
        [[90, 110], [170, 108], [130, 140], [110, 172], [150, 172]], dtype=np.float32
    )
    anchors = RetinaFaceAlignmentStage._build_scalp_anchors(
        landmarks=landmarks,
        face_box=(80.0, 80.0, 180.0, 200.0),
        image_shape=(256, 256),
        canonical_anchors=None,
    )
    assert anchors.shape == (14, 2)
    # sideburns and jaw guides should sit below the cheek anchors.
    assert anchors[7, 1] > anchors[1, 1]
    assert anchors[11, 1] > anchors[9, 1]


def test_extraction_stabilization_fills_interior_holes() -> None:
    stage = ModNetExtractionStage.__new__(ModNetExtractionStage)
    stage._settings = _AlphaSettings()
    alpha = np.zeros((60, 60), dtype=np.float32)
    alpha[12:48, 12:48] = 0.82
    alpha[28:32, 28:32] = 0.10

    stabilized = stage._stabilize_hair_alpha(alpha)
    assert stabilized[30, 30] > 0.56
    assert stabilized[20, 20] >= 0.72


def test_warp_stabilization_preserves_confident_core() -> None:
    stage = TpsWarpStage.__new__(TpsWarpStage)
    stage._settings = _AlphaSettings()
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[8:32, 8:32] = 0.9
    alpha[20, 20] = 0.62

    stabilized = stage._stabilize_warped_alpha(alpha)
    assert stabilized[12, 12] >= 0.88
    assert stabilized[20, 20] >= 0.56


def test_blend_prepare_alpha_keeps_strong_core_opaque() -> None:
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[10:30, 10:30] = 0.95
    prepared = LaplacianBlendStage._prepare_alpha(alpha)
    assert prepared[20, 20] >= 0.98


def test_blend_prepare_alpha_does_not_expand_weak_halo() -> None:
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[12:28, 12:28] = 0.9
    alpha[8:32, 8:32] = np.maximum(alpha[8:32, 8:32], 0.18)

    prepared = LaplacianBlendStage._prepare_alpha(alpha)
    assert prepared[8, 20] <= 0.18
    assert prepared[20, 20] >= 0.84

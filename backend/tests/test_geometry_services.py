import numpy as np

from app.services.cranial_hull import CranialHullEstimator
from app.services.landmarks import DenseLandmarkService


def test_dense_landmarks_outputs_expected_shapes() -> None:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    box = (64.0, 64.0, 192.0, 192.0)
    points5 = np.array(
        [[100, 120], [156, 118], [128, 140], [108, 164], [148, 164]], dtype=np.float32
    )
    svc = DenseLandmarkService()
    out = svc.estimate(rgb, box, points5)
    assert out.points.shape == (68, 2)
    assert out.normalized_points.min() >= 0.0
    assert out.normalized_points.max() <= 1.0


def test_cranial_hull_has_non_empty_mask_and_anchors() -> None:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    box = (64.0, 64.0, 192.0, 192.0)
    points5 = np.array(
        [[100, 120], [156, 118], [128, 140], [108, 164], [148, 164]], dtype=np.float32
    )
    dense = DenseLandmarkService().estimate(rgb, box, points5)
    hull = CranialHullEstimator().estimate(rgb.shape[:2], dense)
    assert hull.hull_mask.sum() > 0
    assert hull.canonical_anchors.shape[0] >= 8

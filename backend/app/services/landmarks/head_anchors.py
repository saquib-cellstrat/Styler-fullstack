"""Build head-warp correspondence anchors from the dense facial mesh.

The anchors drive the donor->base hair warp. Quality hinges on using real,
semantically matched points: the full face-oval contour fixes where hair meets
skin (hairline, temples, jaw, sides), and an extrapolated scalp dome gives the
warp control above the hairline where the mesh has no vertices. Donor and base
use identical indices and the identical extrapolation rule, so every anchor
corresponds exactly between the two faces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Canonical face-oval loop (forehead-top -> right -> chin -> left -> back).
_FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)
# Upper arc of the oval: left temple, over the forehead, to right temple.
_UPPER_ARC: tuple[int, ...] = (21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251)

_FOREHEAD_TOP = 10
_CHIN = 152
_LEFT_IRIS = 468
_RIGHT_IRIS = 473
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263


def _eye_centers(pts: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if pts.shape[0] > _RIGHT_IRIS:
        return pts[_LEFT_IRIS], pts[_RIGHT_IRIS]
    return pts[_LEFT_EYE_OUTER], pts[_RIGHT_EYE_OUTER]


def build_head_anchors(
    mesh_points: NDArray[np.float32],
    image_shape: tuple[int, int],
) -> NDArray[np.float32]:
    """Return ordered (N, 2) anchors: face oval + scalp dome bands + crown apex."""
    h, w = image_shape
    pts = mesh_points.astype(np.float32)

    forehead_top = pts[_FOREHEAD_TOP]
    chin = pts[_CHIN]
    left_eye, right_eye = _eye_centers(pts)
    eye_mid = (left_eye + right_eye) / 2.0

    up = forehead_top - chin
    face_height = float(np.linalg.norm(up)) or 1.0
    up = up / face_height

    crown_center = forehead_top + up * (0.55 * face_height)

    oval = pts[list(_FACE_OVAL)]
    upper = pts[list(_UPPER_ARC)]

    # Lower scalp band: lift the hairline arc straight up toward the crown.
    band_low = upper + up * (0.26 * face_height)
    # Upper scalp band: lift further and contract horizontally toward the
    # crown center so the dome narrows instead of ballooning.
    contracted = crown_center + (upper - crown_center) * 0.55
    band_high = contracted + up * (0.34 * face_height)

    apex = crown_center[None, :]

    anchors = np.vstack([oval, band_low, band_high, apex]).astype(np.float32)
    anchors[:, 0] = np.clip(anchors[:, 0], 0.0, float(w - 1))
    anchors[:, 1] = np.clip(anchors[:, 1], 0.0, float(h - 1))
    return anchors

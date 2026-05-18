"""Rigid Moving Least Squares (Schaefer et al., 2006) backward maps for image warping.

Donor anchors p_i map to base anchors q_i. For each base pixel u we recover a donor
coordinate v with fixed-point inversion of the forward map f(v) = q* + R(v - p*).
Maps are built on a coarse grid and upsampled for speed.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def _weights(v: NDArray[np.float64], p: NDArray[np.float32], alpha: float, eps: float) -> NDArray[np.float64]:
    diff = p.astype(np.float64) - v
    d2 = np.sum(diff * diff, axis=1)
    d2 = np.maximum(d2, eps * eps)
    if abs(alpha - 1.0) < 1e-9:
        return 1.0 / d2
    return np.power(d2, -alpha)


def _rigid_mls_step(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    p: NDArray[np.float32],
    q: NDArray[np.float32],
    alpha: float,
    eps: float,
) -> NDArray[np.float64]:
    w = _weights(v, p, alpha, eps)
    sw = float(np.sum(w)) + 1e-12
    p_star = (w @ p.astype(np.float64)) / sw
    q_star = (w @ q.astype(np.float64)) / sw
    ph = p.astype(np.float64) - p_star
    qh = q.astype(np.float64) - q_star
    m = (qh * w[:, None]).T @ ph
    u_mat, _, vt = np.linalg.svd(m, full_matrices=True)
    r = u_mat @ vt
    if float(np.linalg.det(r)) < 0.0:
        u_mat = u_mat.copy()
        u_mat[:, 1] *= -1.0
        r = u_mat @ vt
    return p_star + r.T @ (u - q_star)


def inverse_rigid_mls_sample(
    u: NDArray[np.float64],
    p: NDArray[np.float32],
    q: NDArray[np.float32],
    *,
    max_iterations: int,
    alpha: float,
    eps: float,
    tol: float = 1e-3,
) -> NDArray[np.float64]:
    v = u.copy()
    for _ in range(max_iterations):
        v_new = _rigid_mls_step(u, v, p, q, alpha, eps)
        if float(np.linalg.norm(v_new - v)) < tol:
            return v_new
        v = v_new
    return v


def build_backward_rigid_mls_remap_maps(
    donor_anchors: NDArray[np.float32],
    base_anchors: NDArray[np.float32],
    output_shape: tuple[int, int],
    donor_shape: tuple[int, int],
    *,
    grid_long_edge: int,
    max_iterations: int,
    alpha: float,
    eps: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return map_x, map_y for cv2.remap (destination = base, source = donor).

    For each base pixel u, samples donor at v ≈ f^{-1}(u) where f is rigid MLS
    from donor to base driven by pairs (p_i -> q_i).
    """
    h, w = output_shape
    dh, dw = donor_shape
    max_dim = max(float(h), float(w))
    gw = max(4, int(round(grid_long_edge * w / max_dim)))
    gh = max(4, int(round(grid_long_edge * h / max_dim)))

    xs = np.linspace(0.0, float(w - 1), gw, dtype=np.float64)
    ys = np.linspace(0.0, float(h - 1), gh, dtype=np.float64)
    coarse_mx = np.zeros((gh, gw), dtype=np.float32)
    coarse_my = np.zeros((gh, gw), dtype=np.float32)

    p = donor_anchors.astype(np.float32)
    q = base_anchors.astype(np.float32)

    for iy in range(gh):
        for ix in range(gw):
            u = np.array([xs[ix], ys[iy]], dtype=np.float64)
            v = inverse_rigid_mls_sample(u, p, q, max_iterations=max_iterations, alpha=alpha, eps=eps)
            coarse_mx[iy, ix] = float(np.clip(v[0], 0.0, float(dw - 1)))
            coarse_my[iy, ix] = float(np.clip(v[1], 0.0, float(dh - 1)))

    map_x = cv2.resize(coarse_mx, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    map_y = cv2.resize(coarse_my, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return map_x, map_y

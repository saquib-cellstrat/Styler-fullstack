"""ExtractionService tests using stub ONNX sessions.

These tests exercise the real preprocessing/postprocessing path while
substituting MODNet and RetinaFace with deterministic in-process stubs,
so they don't require any ONNX weight files on disk.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.core.model_registry import InferenceRegistry, LoadedOnnxModel
from app.core.settings import get_settings
from app.services.extraction.extraction_service import ExtractionService
from app.services.extraction.quality_gate import (
    NoFaceFoundError,
    NonFrontalFaceError,
)


class _StubModnetSession:
    output_meta = ("alpha",)

    def get_inputs(self) -> list[object]:
        return [type("I", (), {"name": "input", "shape": [1, 3, 512, 512]})()]

    def get_outputs(self) -> list[object]:
        return [type("O", (), {"name": "alpha", "shape": [1, 1, 512, 512]})()]

    def run(self, output_names, feed) -> list[np.ndarray]:
        h, w = 512, 512
        alpha = np.zeros((1, 1, h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        radius = h // 4
        ys, xs = np.ogrid[:h, :w]
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2
        alpha[0, 0][mask] = 1.0
        return [alpha]


class _StubRetinaFaceSession:
    """Returns one centered, perfectly-frontal face."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def get_inputs(self) -> list[object]:
        return [type("I", (), {"name": "input0", "shape": [1, 3, 640, 640]})()]

    def get_outputs(self) -> list[object]:
        return [
            type("O", (), {"name": "loc", "shape": [1, -1, 4]})(),
            type("O", (), {"name": "conf", "shape": [1, -1, 2]})(),
            type("O", (), {"name": "land", "shape": [1, -1, 10]})(),
        ]

    def run(self, output_names, feed):
        from itertools import product
        from math import ceil

        s = get_settings()
        steps = s.retinaface_steps
        min_sizes = s.retinaface_min_sizes
        anchors: list[list[float]] = []
        for k, step in enumerate(steps):
            fh = ceil(self.image_size / step)
            fw = ceil(self.image_size / step)
            for i, j in product(range(fh), range(fw)):
                for ms in min_sizes[k]:
                    cx = (j + 0.5) * step / self.image_size
                    cy = (i + 0.5) * step / self.image_size
                    sk = ms / self.image_size
                    anchors.append([cx, cy, sk, sk])
        priors = np.asarray(anchors, dtype=np.float32)
        n = len(priors)

        loc = np.zeros((1, n, 4), dtype=np.float32)
        landms = np.zeros((1, n, 10), dtype=np.float32)

        # Pick the prior closest to the image center, force its decoded box
        # to span 60% of the image and place 5 landmarks symmetrically.
        center = np.array([0.5, 0.5], dtype=np.float32)
        center_idx = int(np.argmin(np.linalg.norm(priors[:, :2] - center, axis=1)))

        v0, v1 = s.retinaface_variances
        prior = priors[center_idx]
        target_w = target_h = 0.6
        target_cx = target_cy = 0.5
        loc[0, center_idx, 0] = (target_cx - prior[0]) / (v0 * prior[2])
        loc[0, center_idx, 1] = (target_cy - prior[1]) / (v0 * prior[3])
        loc[0, center_idx, 2] = np.log(target_w / prior[2]) / v1
        loc[0, center_idx, 3] = np.log(target_h / prior[3]) / v1

        landmark_offsets = [
            (-0.10, -0.05),
            (0.10, -0.05),
            (0.0, 0.05),
            (-0.07, 0.15),
            (0.07, 0.15),
        ]
        for k, (dx, dy) in enumerate(landmark_offsets):
            tx = target_cx + dx
            ty = target_cy + dy
            landms[0, center_idx, 2 * k] = (tx - prior[0]) / (v0 * prior[2])
            landms[0, center_idx, 2 * k + 1] = (ty - prior[1]) / (v0 * prior[3])

        conf = np.zeros((1, n, 2), dtype=np.float32)
        conf[0, :, 0] = 5.0
        conf[0, center_idx, 0] = -5.0
        conf[0, center_idx, 1] = 5.0
        return [loc, conf, landms]


def _make_registry() -> InferenceRegistry:
    s = get_settings()
    modnet = LoadedOnnxModel(
        session=_StubModnetSession(),  # type: ignore[arg-type]
        input_names=("input",),
        output_names=("alpha",),
    )
    retinaface = LoadedOnnxModel(
        session=_StubRetinaFaceSession(s.retinaface_input_size),  # type: ignore[arg-type]
        input_names=("input0",),
        output_names=("loc", "conf", "land"),
    )
    return InferenceRegistry(modnet=modnet, retinaface=retinaface)


def _png_bytes(width: int = 512, height: int = 512) -> bytes:
    rgb = np.full((height, width, 3), 200, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_extraction_returns_rgba_with_alpha_matte() -> None:
    service = ExtractionService(registry=_make_registry(), settings=get_settings())
    result = service.process(_png_bytes())
    assert result.rgba.shape[2] == 4
    assert result.rgba.dtype == np.uint8
    assert result.alpha.min() >= 0.0 and result.alpha.max() <= 1.0
    png = service.encode_png(result)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_extraction_rejects_empty_payload() -> None:
    service = ExtractionService(registry=_make_registry(), settings=get_settings())
    with pytest.raises(ValueError):
        service.process(b"")


def test_quality_gate_rejects_nonfrontal_face() -> None:
    """Force a heavily off-center nose -> large estimated yaw."""

    class TurnedFaceSession(_StubRetinaFaceSession):
        def run(self, output_names, feed):
            outputs = super().run(output_names, feed)
            loc, conf, landms = outputs
            s = get_settings()
            v0, _ = s.retinaface_variances
            from math import ceil
            from itertools import product

            anchors: list[list[float]] = []
            for k, step in enumerate(s.retinaface_steps):
                fh = ceil(self.image_size / step)
                fw = ceil(self.image_size / step)
                for i, j in product(range(fh), range(fw)):
                    for ms in s.retinaface_min_sizes[k]:
                        cx = (j + 0.5) * step / self.image_size
                        cy = (i + 0.5) * step / self.image_size
                        sk = ms / self.image_size
                        anchors.append([cx, cy, sk, sk])
            priors = np.asarray(anchors, dtype=np.float32)
            center = np.array([0.5, 0.5], dtype=np.float32)
            idx = int(np.argmin(np.linalg.norm(priors[:, :2] - center, axis=1)))
            prior = priors[idx]
            # Push the nose far to one side relative to the eye midpoint.
            tx = 0.5 + 0.25
            ty = 0.5 + 0.05
            landms[0, idx, 4] = (tx - prior[0]) / (v0 * prior[2])
            landms[0, idx, 5] = (ty - prior[1]) / (v0 * prior[3])
            return [loc, conf, landms]

    s = get_settings()
    registry = _make_registry()
    registry = InferenceRegistry(
        modnet=registry.modnet,
        retinaface=LoadedOnnxModel(
            session=TurnedFaceSession(s.retinaface_input_size),  # type: ignore[arg-type]
            input_names=("input0",),
            output_names=("loc", "conf", "land"),
        ),
    )
    service = ExtractionService(registry=registry, settings=s)
    with pytest.raises(NonFrontalFaceError):
        service.process(_png_bytes())


def test_quality_gate_rejects_when_no_face_detected() -> None:
    class NoFaceSession(_StubRetinaFaceSession):
        def run(self, output_names, feed):
            loc, conf, landms = super().run(output_names, feed)
            conf[0, :, 0] = 5.0
            conf[0, :, 1] = -5.0
            return [loc, conf, landms]

    s = get_settings()
    registry = _make_registry()
    registry = InferenceRegistry(
        modnet=registry.modnet,
        retinaface=LoadedOnnxModel(
            session=NoFaceSession(s.retinaface_input_size),  # type: ignore[arg-type]
            input_names=("input0",),
            output_names=("loc", "conf", "land"),
        ),
    )
    service = ExtractionService(registry=registry, settings=s)
    with pytest.raises(NoFaceFoundError):
        service.process(_png_bytes())

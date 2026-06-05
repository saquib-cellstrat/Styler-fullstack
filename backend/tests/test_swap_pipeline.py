from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.bootstrap import create_app
from app.core.dependencies import get_master_pipeline
from app.pipeline.builder import build_default_registry
from app.pipeline.contracts import HairRegions
from app.pipeline.context import ProcessingContext
from app.pipeline.master import MasterPipeline
from app.pipeline.stages.blending import LaplacianBlendStage
from app.pipeline.stages.harmonization import LabColorTransferStage
from app.pipeline.stages.warping import TpsWarpStage
from app.pipeline.stages.warping_mls import MlsWarpStage


class _StubPipeline:
    def run(
        self,
        context: ProcessingContext,
        implementation_overrides: dict[str, str] | None = None,
    ) -> ProcessingContext:
        _ = implementation_overrides
        output = np.full((64, 64, 4), 255, dtype=np.uint8)
        output[..., :3] = np.array([120, 90, 80], dtype=np.uint8)
        context.output_rgba = output
        context.selected_implementations = {
            "alignment": "retinaface",
            "extraction": "modnet",
            "warp": "tps",
            "harmonization": "lab_transfer",
            "blending": "laplacian",
        }
        context.timings_ms = {"alignment": 1.0, "blending": 2.0}
        return context


def _png_bytes(width: int = 128, height: int = 128) -> bytes:
    rgb = np.full((height, width, 3), 200, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_swap_hair_endpoint_accepts_two_images() -> None:
    app = create_app()
    app.dependency_overrides[get_master_pipeline] = lambda: _StubPipeline()
    client = TestClient(app)
    files = {
        "base_image": ("base.png", _png_bytes(), "image/png"),
        "donor_image": ("donor.png", _png_bytes(), "image/png"),
    }
    response = client.post("/api/v1/swap-hair", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["x-pipeline-warp"] == "tps"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_master_pipeline_executes_registered_stages() -> None:
    class _NoopModels:
        pass

    class _NoopSettings:
        pipeline_alignment_impl = "retinaface"
        pipeline_extraction_impl = "modnet"
        pipeline_warp_impl = "tps"
        pipeline_harmonization_impl = "lab_transfer"
        pipeline_blending_impl = "laplacian"

    class _Stage:
        def __init__(self, name: str) -> None:
            self.name = name

        def process(self, context: ProcessingContext) -> ProcessingContext:
            context.timings_ms[f"visited_{self.name}"] = 1.0
            return context

    registry = build_default_registry()
    registry.register("alignment", "retinaface", lambda _m, _s: _Stage("alignment"))
    registry.register("extraction", "modnet", lambda _m, _s: _Stage("extraction"))
    registry.register("warp", "tps", lambda _m, _s: _Stage("warp"))
    registry.register("harmonization", "lab_transfer", lambda _m, _s: _Stage("harmonization"))
    registry.register("blending", "laplacian", lambda _m, _s: _Stage("blending"))

    pipeline = MasterPipeline(registry=registry, models=_NoopModels(), settings=_NoopSettings())
    context = ProcessingContext(base_image_bytes=b"x", donor_image_bytes=b"y")
    result = pipeline.run(context)
    assert "alignment" in result.selected_implementations
    assert "visited_blending" in result.timings_ms


def test_tps_warp_identity_shape() -> None:
    class _NoopModels:
        pass

    class _Settings:
        tps_regularization = 1e-3
        enable_region_aware_tps = False
        debug_export_enabled = False
        debug_export_dir = "debug/exports"

    stage = TpsWarpStage(models=_NoopModels(), settings=_Settings())
    anchor = np.array(
        [[30, 20], [60, 20], [45, 10], [45, 40], [30, 40], [60, 40], [45, 55]],
        dtype=np.float32,
    )
    rgba = np.zeros((80, 80, 4), dtype=np.uint8)
    rgba[20:60, 30:60, :3] = 255
    alpha = np.zeros((80, 80), dtype=np.float32)
    alpha[20:60, 30:60] = 1.0
    base_rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    context = ProcessingContext(
        base_image_bytes=b"",
        donor_image_bytes=b"",
        base_rgb=base_rgb,
        donor_hair_rgba=rgba,
        donor_hair_alpha=alpha,
        base_scalp_anchors=anchor,
        donor_scalp_anchors=anchor.copy(),
        donor_regions_v2=HairRegions(
            roots=np.ones((80, 80), dtype=np.float32),
            forehead_line=np.zeros((80, 80), dtype=np.float32),
            temples=np.zeros((80, 80), dtype=np.float32),
            side_strands=np.zeros((80, 80), dtype=np.float32),
            long_strands=np.ones((80, 80), dtype=np.float32),
            shoulder_overlap=np.zeros((80, 80), dtype=np.float32),
        ),
    )
    result = stage.process(context)
    assert result.warped_hair_rgb is not None
    assert result.warped_hair_alpha is not None
    assert result.warped_hair_rgb.shape[:2] == base_rgb.shape[:2]


def test_mls_warp_identity_shape() -> None:
    class _NoopModels:
        pass

    class _Settings:
        enable_region_aware_tps = False
        debug_export_enabled = False
        debug_export_dir = "debug/exports"
        mls_map_grid_long_edge = 24
        mls_inverse_max_iterations = 12
        mls_weight_alpha = 1.0
        mls_weight_eps = 2.0

    stage = MlsWarpStage(models=_NoopModels(), settings=_Settings())
    anchor = np.array(
        [[30, 20], [60, 20], [45, 10], [45, 40], [30, 40], [60, 40], [45, 55]],
        dtype=np.float32,
    )
    rgba = np.zeros((80, 80, 4), dtype=np.uint8)
    rgba[20:60, 30:60, :3] = 255
    alpha = np.zeros((80, 80), dtype=np.float32)
    alpha[20:60, 30:60] = 1.0
    base_rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    context = ProcessingContext(
        base_image_bytes=b"",
        donor_image_bytes=b"",
        base_rgb=base_rgb,
        donor_hair_rgba=rgba,
        donor_hair_alpha=alpha,
        base_scalp_anchors=anchor,
        donor_scalp_anchors=anchor.copy(),
    )
    result = stage.process(context)
    assert result.warped_hair_rgb is not None
    assert result.warped_hair_alpha is not None
    assert result.warped_hair_rgb.shape[:2] == base_rgb.shape[:2]


def test_harmonization_moves_hair_luminance_toward_base_reference() -> None:
    class _NoopModels:
        pass

    class _Settings:
        harmonization_alpha_threshold = 0.12
        harmonization_ring_kernel = 9
        harmonization_luma_strength = 1.0
        harmonization_contrast_strength = 0.0
        harmonization_chroma_strength = 0.0
        harmonization_max_luma_shift = 80.0
        harmonization_max_chroma_shift = 0.0
        harmonization_min_pixels = 16

    base_rgb = np.full((64, 64, 3), 70, dtype=np.uint8)
    hair_rgb = np.full((64, 64, 3), 220, dtype=np.uint8)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[20:44, 20:44] = 1.0
    context = ProcessingContext(
        base_image_bytes=b"",
        donor_image_bytes=b"",
        base_rgb=base_rgb,
        warped_hair_rgb=hair_rgb,
        warped_hair_alpha=alpha,
    )

    result = LabColorTransferStage(_NoopModels(), _Settings()).process(context)

    assert result.harmonized_hair_rgb is not None
    original_mean = float(hair_rgb[alpha > 0.5].mean())
    harmonized_mean = float(result.harmonized_hair_rgb[alpha > 0.5].mean())
    assert harmonized_mean < original_mean


def test_blending_solidifies_transparent_edge_rgb() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[12:20, 12:20] = np.array([180, 60, 40], dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[12:20, 12:20] = 1.0
    alpha[10:22, 10:22] = np.maximum(alpha[10:22, 10:22], 0.10)

    filled = LaplacianBlendStage._solidify_edge_rgb(rgb, alpha)

    assert int(filled[10, 16].sum()) > 0

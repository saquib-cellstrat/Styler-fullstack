from pathlib import Path

from app.core.model_registry import ModelSession
from app.services.blending.service import BlendingService


def test_blending_service_returns_stage_payload() -> None:
    service = BlendingService(
        model_session=ModelSession("blending", Path("models/weights/blending.onnx"))
    )
    result = service.run([0.2, 0.8])
    assert result["stage"] == "blending"
    assert result["opacity"] == 0.5

from pathlib import Path

from app.core.model_registry import ModelSession
from app.services.warping.service import WarpingService


def test_warping_service_returns_stage_payload() -> None:
    service = WarpingService(
        model_session=ModelSession("warping", Path("models/weights/warping.onnx"))
    )
    result = service.run([0.0, 4.0])
    assert result["stage"] == "warping"
    assert isinstance(result["control_points"], list)

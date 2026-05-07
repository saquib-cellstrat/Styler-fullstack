from app.core.model_registry import ModelSession
from app.shared.math_utils import interpolate


class WarpingService:
    def __init__(self, model_session: ModelSession):
        self.model_session = model_session

    def run(self, control_points: list[float]) -> dict[str, object]:
        if len(control_points) < 2:
            warped = control_points
        else:
            warped = [interpolate(control_points[0], control_points[-1], 0.5)]
        return {
            "stage": "warping",
            "model": str(self.model_session.weights_path),
            "control_points": warped,
        }

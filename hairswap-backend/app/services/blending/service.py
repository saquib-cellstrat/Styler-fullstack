from app.core.model_registry import ModelSession


class BlendingService:
    def __init__(self, model_session: ModelSession):
        self.model_session = model_session

    def run(self, alpha_map: list[float]) -> dict[str, object]:
        opacity = sum(alpha_map) / len(alpha_map) if alpha_map else 0.0
        return {
            "stage": "blending",
            "model": str(self.model_session.weights_path),
            "opacity": opacity,
        }

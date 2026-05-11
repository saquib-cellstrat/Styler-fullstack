from pathlib import Path

from app.core.model_registry import InferenceRegistry, LoadedOnnxModel
from app.core.onnx_session import create_inference_session, wrap_loaded_model
from app.core.settings import Settings, get_settings


class ModelManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._registry: InferenceRegistry | None = None

    def load(self) -> InferenceRegistry:
        if self._registry is not None:
            return self._registry

        weights_dir = Path(self.settings.model_weights_dir)
        face_parser = self._load_required(weights_dir / self.settings.face_parser_onnx_filename)
        retinaface = self._load_required(weights_dir / self.settings.retinaface_onnx_filename)
        modnet = self._load_optional(weights_dir / self.settings.modnet_onnx_filename)
        self._registry = InferenceRegistry(
            face_parser=face_parser,
            retinaface=retinaface,
            modnet=modnet,
        )
        return self._registry

    def get_registry(self) -> InferenceRegistry | None:
        return self._registry

    def clear(self) -> None:
        self._registry = None

    def _load_required(self, path: Path) -> LoadedOnnxModel:
        if not path.exists():
            raise FileNotFoundError(f"Required model weights not found: {path}")
        session = create_inference_session(path, self.settings)
        return wrap_loaded_model(session)

    def _load_optional(self, path: Path) -> LoadedOnnxModel | None:
        if not path.exists():
            return None
        session = create_inference_session(path, self.settings)
        return wrap_loaded_model(session)

from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort


@dataclass(frozen=True)
class ModelSession:
    """Lightweight, weight-path-only handle used by the legacy warp/blend stubs."""

    model_name: str
    weights_path: Path


@dataclass(frozen=True)
class LoadedOnnxModel:
    """Wraps a loaded ONNX Runtime session plus resolved metadata."""

    session: ort.InferenceSession
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class InferenceRegistry:
    """All inference sessions needed for the hair extraction pipeline."""

    modnet: LoadedOnnxModel
    retinaface: LoadedOnnxModel

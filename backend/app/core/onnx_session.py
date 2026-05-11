from pathlib import Path

import onnxruntime as ort

from app.core.model_registry import LoadedOnnxModel
from app.core.settings import Settings


def create_inference_session(model_path: Path, settings: Settings) -> ort.InferenceSession:
    """Build an ONNX Runtime session using configured execution providers."""
    available = set(ort.get_available_providers())
    providers: list[str | tuple[str, dict[str, object]]] = []
    for name in settings.ort_provider_priority:
        if name in available:
            providers.append(name)
    if not providers:
        providers = ["CPUExecutionProvider"]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=providers,
    )


def wrap_loaded_model(session: ort.InferenceSession) -> LoadedOnnxModel:
    inputs = tuple(o.name for o in session.get_inputs())
    outputs = tuple(o.name for o in session.get_outputs())
    return LoadedOnnxModel(session=session, input_names=inputs, output_names=outputs)


def resolve_modnet_output_name(
    loaded: LoadedOnnxModel,
    configured_name: str,
) -> str:
    if configured_name and configured_name in loaded.output_names:
        return configured_name
    for out in loaded.session.get_outputs():
        shape = out.shape
        flat = [s for s in shape if isinstance(s, int)]
        if len(flat) >= 2 and flat[-2] == flat[-1]:
            return out.name
    return loaded.output_names[-1]

from pathlib import Path

from fastapi import Depends, HTTPException, Request, status

from app.core.model_manager import ModelManager
from app.core.model_registry import InferenceRegistry, ModelSession
from app.core.onnx_session import (
    create_inference_session,
    wrap_loaded_model,
)
from app.core.settings import Settings, get_settings
from app.pipeline.builder import build_master_pipeline
from app.pipeline.master import MasterPipeline
from app.services.blending.service import BlendingService
from app.services.extraction.extraction_service import ExtractionService
from app.services.warping.service import WarpingService


def load_inference_registry() -> InferenceRegistry:
    """Build all ONNX Runtime sessions once at app startup."""
    settings = get_settings()
    weights_dir = Path(settings.model_weights_dir)
    face_parser_path = weights_dir / settings.face_parser_onnx_filename
    retinaface_path = weights_dir / settings.retinaface_onnx_filename
    modnet_path = weights_dir / settings.modnet_onnx_filename

    if not face_parser_path.exists():
        raise FileNotFoundError(
            f"Face parser weights not found at {face_parser_path}. "
            f"Place the BiSeNet ONNX file under {weights_dir}/."
        )
    if not retinaface_path.exists():
        raise FileNotFoundError(
            f"RetinaFace weights not found at {retinaface_path}. "
            f"Place the ONNX file under {weights_dir}/."
        )

    face_parser = wrap_loaded_model(
        create_inference_session(face_parser_path, settings)
    )
    retinaface = wrap_loaded_model(
        create_inference_session(retinaface_path, settings)
    )
    modnet = (
        wrap_loaded_model(create_inference_session(modnet_path, settings))
        if modnet_path.exists()
        else None
    )
    return InferenceRegistry(
        face_parser=face_parser,
        retinaface=retinaface,
        modnet=modnet,
    )


def warmup_model_sessions() -> dict[str, ModelSession]:
    """Stub model-session map used by the legacy warp/blend endpoints."""
    settings = get_settings()
    weights_dir = Path(settings.model_weights_dir)
    return {
        "warping": ModelSession("warping", weights_dir / "warping.onnx"),
        "blending": ModelSession("blending", weights_dir / "blending.onnx"),
    }


def get_inference_registry(request: Request) -> InferenceRegistry:
    registry = getattr(request.app.state, "inference_registry", None)
    if registry is None:
        model_manager: ModelManager | None = getattr(request.app.state, "model_manager", None)
        if model_manager is not None:
            registry = model_manager.get_registry()
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Inference models are not loaded. "
                "Provision MODNet + RetinaFace ONNX weights and restart the service."
            ),
        )
    return registry


def get_model_sessions(request: Request) -> dict[str, ModelSession]:
    return getattr(request.app.state, "model_sessions", {})


def get_app_settings() -> Settings:
    return get_settings()


def get_extraction_service(
    registry: InferenceRegistry = Depends(get_inference_registry),
    settings: Settings = Depends(get_app_settings),
) -> ExtractionService:
    return ExtractionService(registry=registry, settings=settings)


def get_warping_service(
    sessions: dict[str, ModelSession] = Depends(get_model_sessions),
) -> WarpingService:
    return WarpingService(model_session=sessions["warping"])


def get_blending_service(
    sessions: dict[str, ModelSession] = Depends(get_model_sessions),
) -> BlendingService:
    return BlendingService(model_session=sessions["blending"])


def get_master_pipeline(
    registry: InferenceRegistry = Depends(get_inference_registry),
    settings: Settings = Depends(get_app_settings),
) -> MasterPipeline:
    return build_master_pipeline(models=registry, settings=settings)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router_registry import include_feature_routers
from app.api.v1.router import include_v1_endpoints
from app.core.dependencies import warmup_model_sessions
from app.core.model_manager import ModelManager
from app.core.settings import get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_sessions = warmup_model_sessions()
    app.state.model_manager = ModelManager(get_settings())
    try:
        app.state.inference_registry = app.state.model_manager.load()
    except FileNotFoundError as exc:
        log.warning("Inference registry not initialized: %s", exc)
        app.state.inference_registry = None
    try:
        yield
    finally:
        app.state.inference_registry = None
        app.state.model_manager.clear()
        app.state.model_sessions.clear()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    include_feature_routers(app)
    include_v1_endpoints(app)
    return app

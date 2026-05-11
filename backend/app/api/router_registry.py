from importlib import import_module
from pkgutil import iter_modules

from fastapi import FastAPI

from app.core.settings import get_settings
import app.features as features_package


def include_feature_routers(fastapi_app: FastAPI) -> None:
    settings = get_settings()
    for module_info in iter_modules(features_package.__path__):
        router_module = f"app.features.{module_info.name}.router"
        try:
            module = import_module(router_module)
        except ModuleNotFoundError as exc:
            if exc.name != router_module:
                raise
            continue

        router = getattr(module, "router", None)
        if router is not None:
            fastapi_app.include_router(router, prefix=settings.api_prefix)

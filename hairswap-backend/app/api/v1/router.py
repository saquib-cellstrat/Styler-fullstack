"""Aggregates all v1 endpoint routers under the configured API prefix."""

from fastapi import FastAPI

from app.api.v1.endpoints import extraction
from app.core.settings import get_settings


def include_v1_endpoints(app: FastAPI) -> None:
    settings = get_settings()
    app.include_router(extraction.router, prefix=settings.api_prefix)

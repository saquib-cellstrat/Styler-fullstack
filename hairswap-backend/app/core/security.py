from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.settings import get_settings

settings = get_settings()
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key

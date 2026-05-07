"""POST /extract-hair: high-fidelity RGBA hair extraction."""

from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_extraction_service
from app.core.security import verify_api_key
from app.services.extraction import (
    ExtractionService,
    NoFaceFoundError,
    NonFrontalFaceError,
)


router = APIRouter(tags=["extraction"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@router.post(
    "/extract-hair",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"image/png": {}}},
        400: {"description": "Malformed image payload"},
        415: {"description": "Unsupported image format"},
        422: {"description": "Quality gate rejected the donor image"},
        503: {"description": "Inference models are not loaded"},
    },
)
async def extract_hair(
    image: UploadFile = File(..., description="Donor portrait image"),
    service: ExtractionService = Depends(get_extraction_service),
    _: str = Depends(verify_api_key),
) -> StreamingResponse:
    if image.content_type and image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {image.content_type}",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload payload",
        )

    try:
        result = service.process(raw)
    except NoFaceFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except NonFrontalFaceError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    png_bytes = service.encode_png(result)
    headers = {
        "Content-Disposition": 'inline; filename="hair-extraction.png"',
        "X-Face-Yaw-Deg": f"{result.face.estimated_yaw_deg:.2f}",
        "X-Face-Score": f"{result.face.score:.3f}",
    }
    return StreamingResponse(
        BytesIO(png_bytes),
        media_type="image/png",
        headers=headers,
    )

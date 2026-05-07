"""POST /swap-hair: multi-stage donor-to-base hair transfer pipeline."""

from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_master_pipeline
from app.core.security import verify_api_key
from app.pipeline.context import ProcessingContext
from app.pipeline.master import MasterPipeline
from app.shared.image_processing import ImageProcessor

router = APIRouter(tags=["pipeline"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@router.post(
    "/swap-hair",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"image/png": {}}},
        400: {"description": "Malformed image payload"},
        415: {"description": "Unsupported image format"},
        422: {"description": "Pipeline stage failed"},
        503: {"description": "Inference models are not loaded"},
    },
)
async def swap_hair(
    base_image: UploadFile = File(..., description="Base bald user image"),
    donor_image: UploadFile = File(..., description="Celebrity/donor image"),
    pipeline: MasterPipeline = Depends(get_master_pipeline),
    _: str = Depends(verify_api_key),
) -> StreamingResponse:
    _validate_content_type(base_image)
    _validate_content_type(donor_image)

    base_raw = await base_image.read()
    donor_raw = await donor_image.read()
    if not base_raw or not donor_raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both base_image and donor_image must be non-empty uploads",
        )

    context = ProcessingContext(base_image_bytes=base_raw, donor_image_bytes=donor_raw)
    try:
        result = pipeline.run(context=context)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if result.output_rgba is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Pipeline did not produce an output image",
        )

    image_processor = ImageProcessor()
    png_bytes = image_processor.encode_png(result.output_rgba, compress_level=6, optimize=False)
    stage_timings = {
        "alignment": result.timings_ms.get("alignment", 0.0),
        "extraction": result.timings_ms.get("extraction", 0.0),
        "warp": result.timings_ms.get("warp", 0.0),
        "harmonization": result.timings_ms.get("harmonization", 0.0),
        "blending": result.timings_ms.get("blending", 0.0),
    }
    total_ms = sum(stage_timings.values())
    headers = {
        "Content-Disposition": 'inline; filename="hair-swap.png"',
        "X-Pipeline-Alignment": result.selected_implementations.get("alignment", ""),
        "X-Pipeline-Extraction": result.selected_implementations.get("extraction", ""),
        "X-Pipeline-Warp": result.selected_implementations.get("warp", ""),
        "X-Pipeline-Harmonization": result.selected_implementations.get("harmonization", ""),
        "X-Pipeline-Blending": result.selected_implementations.get("blending", ""),
        "X-Pipeline-Ms-Alignment": f"{stage_timings['alignment']:.1f}",
        "X-Pipeline-Ms-Extraction": f"{stage_timings['extraction']:.1f}",
        "X-Pipeline-Ms-Warp": f"{stage_timings['warp']:.1f}",
        "X-Pipeline-Ms-Harmonization": f"{stage_timings['harmonization']:.1f}",
        "X-Pipeline-Ms-Blending": f"{stage_timings['blending']:.1f}",
        "X-Pipeline-Ms-Total": f"{total_ms:.1f}",
        "X-Pipeline-Timings-Ms": ",".join(
            f"{name}:{value:.1f}" for name, value in result.timings_ms.items()
        ),
    }
    return StreamingResponse(BytesIO(png_bytes), media_type="image/png", headers=headers)


def _validate_content_type(image: UploadFile) -> None:
    if image.content_type and image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {image.content_type}",
        )

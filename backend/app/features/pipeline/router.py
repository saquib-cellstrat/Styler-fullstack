from fastapi import APIRouter, Depends

from app.core.dependencies import (
    get_blending_service,
    get_warping_service,
)
from app.core.security import verify_api_key
from app.schemas.pipeline import (
    BlendingRequest,
    PipelineStatusResponse,
    StageResponse,
    WarpingRequest,
)
from app.services.blending.service import BlendingService
from app.services.warping.service import WarpingService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status", response_model=PipelineStatusResponse)
def pipeline_status(_: str = Depends(verify_api_key)) -> PipelineStatusResponse:
    return PipelineStatusResponse(status="ok")


@router.post("/warp", response_model=StageResponse)
def warp(
    payload: WarpingRequest,
    service: WarpingService = Depends(get_warping_service),
    _: str = Depends(verify_api_key),
) -> StageResponse:
    return StageResponse(**service.run(payload.control_points))


@router.post("/blend", response_model=StageResponse)
def blend(
    payload: BlendingRequest,
    service: BlendingService = Depends(get_blending_service),
    _: str = Depends(verify_api_key),
) -> StageResponse:
    return StageResponse(**service.run(payload.alpha_map))

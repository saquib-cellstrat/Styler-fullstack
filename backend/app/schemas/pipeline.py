from pydantic import BaseModel, Field


class WarpingRequest(BaseModel):
    control_points: list[float] = Field(default_factory=list)


class BlendingRequest(BaseModel):
    alpha_map: list[float] = Field(default_factory=list)


class StageResponse(BaseModel):
    stage: str
    model: str
    control_points: list[float] | None = None
    opacity: float | None = None


class PipelineStatusResponse(BaseModel):
    status: str

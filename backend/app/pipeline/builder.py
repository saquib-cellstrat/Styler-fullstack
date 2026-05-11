from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.master import MasterPipeline
from app.pipeline.registry import PipelineRegistry
from app.pipeline.stages.alignment import RetinaFaceAlignmentStage
from app.pipeline.stages.blending import LaplacianBlendStage
from app.pipeline.stages.extraction import ModNetExtractionStage
from app.pipeline.stages.harmonization import LabColorTransferStage
from app.pipeline.stages.warping import TpsWarpStage


def build_default_registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register("alignment", "retinaface", RetinaFaceAlignmentStage)
    registry.register("extraction", "modnet", ModNetExtractionStage)
    registry.register("warp", "tps", TpsWarpStage)
    registry.register("harmonization", "lab_transfer", LabColorTransferStage)
    registry.register("blending", "laplacian", LaplacianBlendStage)
    return registry


def build_master_pipeline(models: InferenceRegistry, settings: Settings) -> MasterPipeline:
    return MasterPipeline(registry=build_default_registry(), models=models, settings=settings)

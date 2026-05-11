import time
from dataclasses import dataclass

from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.registry import PipelineRegistry
from app.pipeline.stages.base import AbstractPipelineStage

_STAGE_ORDER: tuple[str, ...] = (
    "alignment",
    "extraction",
    "warp",
    "harmonization",
    "blending",
)


@dataclass(frozen=True)
class MasterPipeline:
    registry: PipelineRegistry
    models: InferenceRegistry
    settings: Settings

    def run(
        self,
        context: ProcessingContext,
        implementation_overrides: dict[str, str] | None = None,
    ) -> ProcessingContext:
        selected = self._resolve_stage_implementations(implementation_overrides)
        context.selected_implementations.update(selected)
        for stage_name in _STAGE_ORDER:
            stage = self.registry.create_stage(
                stage_name,
                selected[stage_name],
                self.models,
                self.settings,
            )
            context = self._run_stage(stage, context)
        return context

    def _resolve_stage_implementations(
        self, implementation_overrides: dict[str, str] | None
    ) -> dict[str, str]:
        defaults = {
            "alignment": self.settings.pipeline_alignment_impl,
            "extraction": self.settings.pipeline_extraction_impl,
            "warp": self.settings.pipeline_warp_impl,
            "harmonization": self.settings.pipeline_harmonization_impl,
            "blending": self.settings.pipeline_blending_impl,
        }
        if implementation_overrides is None:
            return defaults
        merged = defaults.copy()
        for stage_name, impl_name in implementation_overrides.items():
            if impl_name:
                merged[stage_name] = impl_name
        return merged

    @staticmethod
    def _run_stage(
        stage: AbstractPipelineStage, context: ProcessingContext
    ) -> ProcessingContext:
        start = time.perf_counter()
        updated = stage.process(context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        updated.timings_ms[stage.name] = elapsed_ms
        return updated

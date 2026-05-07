from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.stages.base import AbstractPipelineStage

StageFactory = Callable[[InferenceRegistry, Settings], AbstractPipelineStage]


@dataclass
class PipelineRegistry:
    _factories: dict[str, dict[str, StageFactory]] = field(default_factory=dict)

    def register(self, stage_type: str, implementation: str, factory: StageFactory) -> None:
        stage_factories = self._factories.setdefault(stage_type, {})
        stage_factories[implementation] = factory

    def create_stage(
        self,
        stage_type: str,
        implementation: str,
        models: InferenceRegistry,
        settings: Settings,
    ) -> AbstractPipelineStage:
        stage_factories = self._factories.get(stage_type, {})
        if implementation not in stage_factories:
            available = ", ".join(sorted(stage_factories.keys())) or "none"
            raise ValueError(
                f"Unknown implementation '{implementation}' for stage '{stage_type}'. "
                f"Available: {available}"
            )
        return stage_factories[implementation](models, settings)

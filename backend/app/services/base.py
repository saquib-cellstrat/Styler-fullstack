from abc import ABC, abstractmethod
from typing import Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PipelineStage(ABC, Generic[InputT, OutputT]):
    """A single, composable stage of the hair-swap pipeline.

    Each stage consumes a typed input artefact and produces a typed output
    so a future ``MasterPipeline`` can chain stages (extract -> warp -> blend)
    without coupling the API layer to any concrete implementation.
    """

    name: str = ""

    @abstractmethod
    def process(self, payload: InputT) -> OutputT:
        """Run the stage's core transformation."""

    def __call__(self, payload: InputT) -> OutputT:
        return self.process(payload)

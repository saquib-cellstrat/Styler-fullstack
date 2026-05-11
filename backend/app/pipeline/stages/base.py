from abc import ABC, abstractmethod

from app.pipeline.context import ProcessingContext


class AbstractPipelineStage(ABC):
    name: str = ""

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        raise NotImplementedError

    def __call__(self, context: ProcessingContext) -> ProcessingContext:
        return self.process(context)

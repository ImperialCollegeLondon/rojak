from abc import ABC, abstractmethod


class PostProcessor[T](ABC):
    @abstractmethod
    def execute(self) -> T: ...

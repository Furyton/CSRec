"""

tagging system -> prefix

create logger_service
create model
create optim
load model state
load optim state
create trainer
train
test

pipeline?

"""

from abc import ABCMeta, abstractmethod

# from torch import nn

class BaseSched(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.mode = 'train'
        pass

    @abstractmethod
    def _create_logger_service(self, prefix: str):
        pass

    @abstractmethod
    def _fit(self):
        pass

    @abstractmethod
    def _evaluate(self):
        pass

    @abstractmethod
    def _finishing(self):
        pass

    @abstractmethod
    def run(self):
        if 'train' in self.mode:
            self._fit()
        if 'test' in self.mode:
            self._evaluate()

        self._finishing()
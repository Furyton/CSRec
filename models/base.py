import torch
import torch.nn as nn
from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, dataset: list, device: str, max_len: int, T: float=1.0):
        super(BaseModel, self).__init__()
        """
        dataset:
            [item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test]

            or

            [item_train, item_valid, item_test, usernum, itemnum]
        """

        self._device = device
        self.n_item = dataset[4]
        self.n_user = dataset[3]
        self.max_len = max_len
        self._temperature = T

    def set_temperature(self, t: float):
        self._temperature = t

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    @torch.no_grad()
    def predict(self, batch):
        pass

    @abstractmethod
    @torch.no_grad()
    def full_sort_predict(self, batch):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        """
        input:
            batch
        return:
            predict logits, predict loss, reg loss, ...
        """
        pass

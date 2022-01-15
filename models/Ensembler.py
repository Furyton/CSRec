import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from models.base import BaseModel

class Ensembler(nn.Module):
    r"""Ensembler is a container that combines a list of models"""

    def __init__(self, model_list: list[BaseModel], predefined_weight: list[float]=None):
        super(Ensembler, self).__init__()

        self.model_list = model_list
        if predefined_weight is not None:
            self.weight = predefined_weight
        
        self.debug = 5

    @classmethod
    def code(cls):
        return 'ensembler'

    def forward(self, batch):
        predict = [ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ]
        weight_list = self.weight or [1. / len(self.model_list)] * len(self.model_list)

        predict = [model.full_sort_predict(batch).softmax(dim=-1) for idx, model in enumerate(self.model_list)]

        w_predict = [_predict * weight_list[idx] for idx, _predict in enumerate(predict)]

        final_predict = sum(w_predict) / len(predict)

        with torch.no_grad():
            # B x N
            if self.debug != 0:
                self.debug -= 1
                logging.debug("raw predict")
                logging.debug(predict)
                logging.debug("raw confidence")
                logging.debug(f"{[pred.max(dim=-1).values for pred in predict]}")

                logging.debug("weighted predict")
                logging.debug(w_predict)

                logging.debug("final predict")
                logging.debug(final_predict)
                logging.debug("final confidence")
                logging.debug(final_predict.max(dim=-1).values)

        return final_predict


    def calculate_loss(self, batch):
        logging.warning("Not implemented yet.")

        raise RuntimeError("Not implemented yet.")

    def predict(self, batch):
        logging.warn("Not implemented yet.")

        raise RuntimeError("Not implemented yet.")
    
    def full_sort_predict(self, batch):
        logging.warn("Not implemented yet.")

        raise RuntimeError("Not implemented yet.")




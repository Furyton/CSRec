import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from models.base import BaseModel

class Ensembler(nn.Module):
    r"""Ensembler is a container that combines a list of models"""

    def __init__(self, device: str, model_list: list[BaseModel], predefined_weight: list[float]=None, temp: float=1.0):
        super(Ensembler, self).__init__()

        self.model_list = model_list
        for model in self.model_list:
            model.set_temperature(temp)
        if predefined_weight is not None:
            self.weight = predefined_weight
        self._device = device
        self.debug = 5

    @classmethod
    def code(cls):
        return 'ensembler'
    def eval(self):
        for model in self.model_list:
            model: BaseModel
            model.eval()
        return super().eval()

    def eval(self):
        for model in self.model_list:
            model: BaseModel
            model.eval()
        return super().eval()

    def forward(self, batch):
        # return self.weighted_mix(batch)

        # return self.early_exit(batch)

        predict = [ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ]
        weight_list = self.weight or [1. / len(self.model_list)] * len(self.model_list)

        # predict = [model.full_sort_predict(batch) for idx, model in enumerate(self.model_list)]

        w_predict = [_predict * weight_list[idx] for idx, _predict in enumerate(predict)]

        final_predict = sum(w_predict) / sum(weight_list)

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

        # return torch.log(final_predict + 1.)
        return final_predict

    @torch.no_grad()
    def early_exit(self, batch):
        threshold = 0.1
        batch_size = batch[0].size(0)

        stacked_pred = torch.stack([ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ], dim=-1)

        cummulate_sum = torch.cumsum(stacked_pred, dim=-1)

        n_model = len(self.model_list)

        confidence = cummulate_sum.max(dim=1).values / (torch.arange(1, n_model + 1, device=self._device)).repeat(batch_size, 1)


        predict = cummulate_sum[torch.arange(batch_size), :, confidence.max(dim=-1).indices]

        if self.debug != 0:
            self.debug -= 1
            logging.debug(f"stacked_pred: {stacked_pred}, size: {stacked_pred.size()}")

            logging.debug(f"cummulate_sum: {cummulate_sum}, size: {cummulate_sum.size()}")

            logging.debug(f"confidence: {confidence}, size: {confidence.size()}")

            logging.debug(f"predict: {predict}, size: {predict.size()}")

        return predict

        # accumulate_predict = [self.model_list[0].full_sort_predict(batch).softmax(dim=-1)]

        # accumulate_predict = self.model_list[0].full_sort_predict(batch).softmax(dim=-1)
        # accumulate_predict: torch.Tensor
        # for model in self.model_list[1:]:
        #     confidence = accumulate_predict.max(dim=-1).values.unsqueeze(-1)
        #     if max(confidence) >= 1 - threshold:
        #         break
        #     predict = model.full_sort_predict(batch).softmax(dim=-1)
        #     accumulate_predict = confidence * accumulate_predict + (1. - confidence) * predict
        
        # return accumulate_predict
    @torch.no_grad()
    def weighted_mix(self, batch):
        predict_list = [ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ]

        predict = torch.stack(predict_list, dim=1)

        confidence = predict.max(dim=-1).values

        weight = confidence / confidence.sum(-1).unsqueeze(-1)

        weighted_predict = (predict * weight.unsqueeze(-1)).sum(1)

        if self.debug != 0:
            self.debug -= 1
            logging.debug(f"predict: {predict}, size: {predict.size()}")

            logging.debug(f"confidence: {confidence}, size: {confidence.size()}")

            logging.debug(f"weight: {weight}, size: {weight.size()}")

            logging.debug(f"weighted_predict: {weighted_predict}, size: {weighted_predict.size()}")

        return weighted_predict


    def calculate_loss(self, batch):
        logging.warning("Not implemented yet.")

        raise RuntimeError("Not implemented yet.")


    def predict(self, batch):
        # seqs, candidates, labels, seq_lens, user = batch
        candidates = batch[1]
        scores = self.forward(batch)  # B x V
        # scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        return scores
    
    def full_sort_predict(self, batch):
        scores= self.forward(batch)  # B x V
        # scores = scores[:, -1, :]  # B x V
        return scores

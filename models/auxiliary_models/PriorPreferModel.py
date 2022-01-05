import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from models.base import BaseModel

class PriorModel(BaseModel):
    r"""PriorModel is to compute the probability of an observed item interaction, 
        given the real user preference. 
        aka h_k(s)[i]=P(\tilda{X}=k|X=i, s),
    Note:
        not consider `s` currently."""

    def __init__(self, args, dataset: list, device: str, max_len: int):
        super(PriorModel, self).__init__(dataset, device, max_len)

        self.hidden_size = args.hidden_size

        self.observed_embedding = nn.Embedding(self.n_item + 1, self.hidden_size, padding_idx=0)
        self.prior_item_embedding = nn.Embedding(self.n_item + 1, self.hidden_size, padding_idx=0)

        self.apply(self._init_weights)

    def _init_weights(m):
        if isinstance(m, nn.Embedding):
            # nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
            nn.init.xavier_uniform_(m.weight)

    @classmethod
    def code(cls):
        return 'prior'

    def forward(self, batch):
        seqs = batch[0] # not used in v1
        labels = batch[1] # B x 1

        observed_part = self.observed_embedding(labels).squeeze(1)

        return torch.matmul(observed_part, self.prior_item_embedding.weight.transpose(0, 1))


    def calculate_loss(self, batch):
        logging.warn("Can't calculate loss on its own.")

        raise RuntimeError("Can't calculate loss on its own")

    def predict(self, batch):
        logging.warn("PriorModel is not used for prediction tasks")

        raise RuntimeError("PriorModel is not used for prediction tasks")
    
    def full_sort_predict(self, batch):
        logging.warn("PriorModel is not used for prediction tasks")

        raise RuntimeError("PriorModel is not used for prediction tasks")




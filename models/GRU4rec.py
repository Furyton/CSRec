import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_

from models.base import BaseModel


def _init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight)
    elif isinstance(module, nn.GRU):
        xavier_uniform_(module.weight_hh_l0)
        xavier_uniform_(module.weight_ih_l0)


class GRU4RecModel(BaseModel):
    def __init__(self, args, dataset, device, max_len):
        super(GRU4RecModel, self).__init__(dataset, device, max_len)

        self.embedding_size = args.embed_size
        self.hidden_size = args.hidden_units
        self.num_layers = args.num_layers
        self.dropout_prob = args.hidden_dropout

        self.loss_type = args.loss_type

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_item + 1, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        else:
            logging.critical(f"{self.loss_type} has not been implemented yet.")
            raise NotImplementedError

        # parameters initialization
        self.apply(_init_weights)

    @classmethod
    def code(cls):
        return 'gru4rec'

    def log2feats(self, x):
        item_seq_emb = self.item_embedding(x)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # indices = torch.tensor([self.max_len] * x.size(0))
        # seq_output = self.gather_indexes(gru_output, indices)
        return gru_output

    def forward(self, batch):
        # seqs, labels, rating, seq_lens, user = batch
        seqs = batch[0]
        seq_lens = batch[3]
        x = self.log2feats(seqs)

        # seq_output =  x[:, -1, :].squeeze(1) # B * D
        seq_output = self.gather_indexes(x, seq_lens - 1)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        return logits

    def calculate_loss(self, batch):
        labels = batch[1]
        cl = labels.squeeze()
        logits = self.forward(batch)
        loss = self.loss_fct(logits, cl)

        return logits, loss

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

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

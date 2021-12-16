
import logging
import torch
import torch.nn as nn

from models.base import BaseModel
from models.bert_modules.embedding.bert import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock


class BERTModel(BaseModel):
    def __init__(self, args, dataset, device, max_len):
        super(BERTModel, self).__init__(dataset, device, max_len)

        vocab_size = self.n_item + 2
        self.n_layers = args.num_blocks
        heads = args.num_heads

        # 2 means [mask] (item_num + 1) and padding (0)
        self.hidden = args.hidden_units
        dropout = args.attention_dropout
        hidden_dropout = args.hidden_dropout
        embedding_size = args.embed_size
        self.loss_type = args.loss_type

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embedding_size=embedding_size, output_size=self.hidden, max_len=self.max_len, dropout=hidden_dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, heads, self.hidden * 4, dropout, hidden_dropout) for _ in range(self.n_layers)])

        self.unidirectional_tf_blocks = None
        self.out = nn.Linear(self.hidden, self.n_item + 1)
        # self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # self.sample_wise_ce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        else:
            logging.critical(f"{self.loss_type} has not been implemented yet.")
            raise NotImplementedError

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, batch):
        # seqs, labels, rating, seq_lens, user = batch
        seqs = batch[0]

        mask = (seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)
        # batch_size x 1 x max_len x max_len
        x = self.embedding(seqs)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return self.out(x)

    # def calculate_sample_wise_loss(self, seqs, labels):
    #     logits, output = self.forward(seqs)  # B x T x V

    #     logits = logits.view(-1, logits.size(-1))  # (B*T) x V

    #     labels = labels.view(-1)  # B*T

    #     loss = self.sample_wise_ce(logits, labels)
        
    #     return loss

    # def calculate_normal_loss(self, seqs, labels):
    #      return self.calculate_sample_wise_loss(seqs, labels).mean()
    
    # def calculate_with_output_embedding(self, seqs, labels):
    #     logits, output = self.forward(seqs)  # B x T x V

    #     logits = logits.view(-1, logits.size(-1))  # (B*T) x V

    #     labels = labels.view(-1)  # B*T

    #     loss = self.sample_wise_ce(logits, labels)
        
    #     return loss, output
    @torch.no_grad() 
    def predict(self, batch):
        # seqs, candidates, labels = batch
        candidates = batch[1]
        scores = self.forward(batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        return scores
        # metrics = recalls_ndcgs_and_mrr_for_ks(scores, labels, self.metric_ks)
        # return metrics

    @torch.no_grad()
    def full_sort_predict(self, batch):
        # seqs, labels = batch
        scores = self.forward(batch)  # B x T x V
        # print("[bert]: scores size" + scores.size())
        scores = scores[:, -1, :].squeeze()  # B x V
        # print("[bert]: after scores size" + scores.size())
        return scores

    def calculate_loss(self, batch):
        labels = batch[1]

        logits = self.forward(batch)

        pred = logits[labels > 0]
        cl = labels[labels > 0]

        loss = self.loss_fct(pred, cl)

        return logits, loss
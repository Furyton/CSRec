import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn import functional as F
from utils import mlp, cos, euclid


class GRU4Rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.device = args.device

        # load parameters info
        self.embedding_size = args.d_model
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_prob = args.dropout

        # define layers and loss
        self.item_embedding = nn.Embedding(args.num_item + 1, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # Output
        self.le_share = args.le_share
        if args.le_share == 0 or args.soft_target == 'unshare':
            self.output = nn.Linear(args.d_model, args.num_item)

        self.soft_target = args.soft_target
        if args.soft_target == 'mlp':
            self.mlp = mlp(args.mlp_hiddent, args.dropout, args.num_item + 1)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def log2feats(self, x):
        item_seq_emb = self.item_embedding(x)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def forward(self, batch):
        item_seq, labels = batch
        x = self.log2feats(item_seq)
        # item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        # gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        # gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        # seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        if self.training:  # 使用这种方式减少现存的占用
            x = x[labels > 0]

        if self.le_share:
            pred = F.linear(x, self.item_embedding.weight[:-1])
        else:
            pred = self.output(x)

        # 在le loss中使用的soft target
        if self.soft_target == 'mlp':
            soft_target = self.mlp(x)
        elif self.soft_target == 'cos':
            soft_target = cos(x, self.item_embedding.weight)
        elif self.soft_target == 'euclid':
            x = x.view(-1, x.size(-1))
            soft_target = euclid(x, self.item_embedding.weight)
            if self.training == False:
                soft_target = soft_target.view(labels.size(0), -1, soft_target.size(-1))
        elif self.soft_target == 'unshare':
            soft_target = self.output(x)
        elif self.soft_target == 'share':
            soft_target = F.linear(x, self.item_embedding.weight)

        return pred, soft_target  # B * L * D --> B * L * N

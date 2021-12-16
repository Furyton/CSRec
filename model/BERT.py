import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mlp, cos, euclid


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x


class BERT(nn.Module):
    """
    BERT model
    i.e., Embbedding + n * TRM + Output
    """

    def __init__(self, args):
        super(BERT, self).__init__()
        self.num = args.num_item
        num_item = args.num_item + 2
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        layers = args.bert_layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        self.device = args.device
        self.embedding_sharing = args.embedding_sharing
        self.model = args.model
        self.max_len = args.max_len
        self.enable_sample = args.enable_sample
        # Embedding
        self.token = nn.Embedding(num_item, d_model)
        self.position = PositionalEmbedding(self.max_len, d_model)

        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)

        # Output
        self.le_share = args.le_share
        if args.le_share == 0 or args.soft_taget == 'unshare':
            self.output = nn.Linear(d_model, num_item - 1)

        self.soft_taget = args.soft_taget
        if args.soft_taget == 'mlp':
            self.mlp = mlp(args.mlp_hiddent, args.dropout, args.num_item + 1)

    def log2feats(self, x):
        # 1 mask
        if self.model == 'bert':
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        else:
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            mask *= self.attention_mask
        # 2 获得特征
        x = self.token(x) + self.position(x)  # B * L --> B * L * D
        for TRM in self.TRMs:
            x = TRM(x, mask)
        return x

    def forward(self, batch):

        x, labels = batch

        x = self.log2feats(x)

        if self.training:  # 使用这种方式减少现存的占用
            x = x[labels > 0]

        if self.le_share:
            pred = F.linear(x, self.token.weight[:-1])
        else:
            pred = self.output(x)

        # 在le loss中使用的soft target
        if self.soft_taget == 'mlp':
            soft_target = self.mlp(x)
        elif self.soft_taget == 'cos':
            soft_target = cos(x, self.token.weight[:-1])
        elif self.soft_taget == 'euclid':
            x = x.view(-1, x.size(-1))
            soft_target = euclid(x, self.token.weight[:-1])
            if not self.training:
                soft_target = soft_target.view(labels.size(0), -1, soft_target.size(-1))
        elif self.soft_taget == 'unshare':
            soft_target = self.output(x)
        elif self.soft_taget == 'share':
            soft_target = F.linear(x, self.token.weight[:-1])

        return pred, soft_target  # B * L * D --> B * L * N

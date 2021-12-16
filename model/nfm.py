import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mlp, cos, euclid


def activation_layer(activation_name='relu'):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        # elif activation_name.lower() == 'dice':
        #     activation = Dice(emb_dim)
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


class NFM(nn.Module):
    def __init__(self, args):
        super(NFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        """
        self.model = args.model
        self.device = args.device

        self.num_features = args.num_item  # 有多少个个item
        self.num_factors = args.d_model  # embeding size是多大
        self.act_function = args.act_function  # mlp中使用什么激活函数
        self.layers = args.nfm_layers  # mlp每层的参数
        self.drop_prob = args.drop_prob  # FM mlp的dropout分别是什么
        self.maxlen = args.max_len
        self.embeddings = nn.Embedding(self.num_features + 1, self.num_factors)
        self.biases = nn.Embedding(self.num_features + 1, 1)  # 每个item的
        self.bias_ = nn.Parameter(torch.tensor([0.0]))
        self.enable_sample = args.enable_sample

        FM_modules = [nn.BatchNorm1d(self.num_factors), nn.Dropout(self.drop_prob[0])]
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = self.num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            MLP_module.append(nn.BatchNorm1d(out_dim))

            MLP_module.append(activation_layer(args.act_function))

            MLP_module.append(nn.Dropout(self.drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)
        # FM的Linear-part
        self.linear_part = nn.Linear(self.maxlen, 1)

        predict_size = self.layers[-1] if self.layers else self.num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        # Output
        self.output = nn.Linear(self.num_factors, self.num_features + 1)
        self.le_share = args.le_share
        if args.le_share == 0 or args.soft_target == 'unshare':
            self.output = nn.Linear(self.num_factors, self.num_features + 1)

        self.soft_taget = args.soft_target
        if args.soft_target == 'mlp':
            self.mlp = mlp(args.mlp_hiddent, args.dropout, self.num_features + 1)

    def log2feats(self, item_seq):  # features [batch_size, seq_len, embed_size]
        # nonzero_embed = self.embeddings(features)
        # feature_values = feature_values.unsqueeze(dim=-1)
        # nonzero_embed = nonzero_embed * feature_values

        features = self.embeddings(item_seq)  # [batch_size, seq_len, embed_size]
        timeline_mask = ~(item_seq == 0).unsqueeze(-1)
        features *= timeline_mask  # broadcast in last dim 将前面的0再次变为0 [batch_size, seq_len, embed_size]

        # Bi-Interaction layer
        sum_square_embed = features.sum(dim=1).pow(2)
        square_sum_embed = (features.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)  # [batch_size, embed_size]
        FM = self.FM_layers(FM)  # [batch_size, embed_size]

        # 这里得到的是FM的二次项 还缺少一次项需要加上
        # [b,maxlen,es]-转置->[b,es,maxlen] -linear->[b,es, 1 ]-转置->[b, 1,es]  es = embedding size 

        linear_part = self.linear_part(features.transpose(2, 1)).reshape(FM.size(0), -1)  # [batch_size, embed_size]
        # FM = linear_part + FM  #[batch_size, embed_size]
        if self.layers:  # have deep layers
            FM = self.deep_layers(FM)

        return linear_part + FM

    def forward(self, batch):

        # 1 数据准备

        item_seq, labels = batch
        # item_seq_emb = self.embeddings(item_seq)  # [batch_size, seq_len, embed_size]
        # timeline_mask = ~(item_seq == 0).unsqueeze(-1)
        # item_seq_emb *= timeline_mask # broadcast in last dim 将前面的0再次变为0 [batch_size, seq_len, embed_size]

        # 2 NFM获得embedding
        x = self.log2feats(item_seq)

        # tmp = self.embeddings.weight.transpose(1, 0)

        if self.le_share:
            pred = F.linear(x, self.embeddings.weight)
        else:
            pred = self.output(x)

        # 在le loss中使用的soft target
        if self.soft_taget == 'mlp':
            soft_target = self.mlp(x)
        elif self.soft_taget == 'cos':
            soft_target = cos(x, self.embeddings.weight)
        elif self.soft_taget == 'euclid':
            x = x.view(-1, x.size(-1))
            soft_target = euclid(x, self.embeddings.weight)
            if self.training == False:
                soft_target = soft_target.view(labels.size(0), -1, soft_target.size(-1))
        elif self.soft_taget == 'unshare':
            soft_target = self.output(x)
        elif self.soft_taget == 'share':
            soft_target = F.linear(x, self.embeddings.weight)

        return pred, soft_target  # B * L * D --> B * L * N

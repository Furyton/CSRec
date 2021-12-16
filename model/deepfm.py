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
        activation = activation_name
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


class MLPLayers(nn.Module):
    r""" MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout, args):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(args.act_function)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class DeepFM(nn.Module):
    def __init__(self, args):
        super(DeepFM, self).__init__()
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
        self.layers = args.dfm_layers  # mlp每层的参数
        self.drop_prob = args.drop_prob  # FM mlp的dropout分别是什么
        self.maxlen = args.max_len
        self.embeddings = nn.Embedding(self.num_features + 1, self.num_factors)
        self.biases = nn.Embedding(self.num_features + 1, 1)  # 每个item的
        self.bias_ = nn.Parameter(torch.tensor([0.0]))
        self.enable_sample = args.enable_sample

        FM_modules = []
        FM_modules.append(nn.BatchNorm1d(self.num_factors))
        FM_modules.append(nn.Dropout(self.drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)
        # FM的Linear-part
        # [b,maxlen,es]-转置->[b,es,maxlen] -linear->[b,es, 1 ]-转置->[b, 1,es] es = embedding size 
        self.linear_part = nn.Linear(self.maxlen, 1)

        # for deep layers
        # 第一种方案 faltten 投入到linear
        in_dim = self.num_factors * args.max_len
        self.layers = [in_dim] + self.layers
        self.mlp_layers = MLPLayers(self.layers, self.drop_prob[-1], args)

        self.sigmoid = nn.Sigmoid()

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
        # [b,maxlen,es]-转置->[b,es,maxlen] -linear->[b,es, 1 ]-转置->[b, 1,es]

        linear_part = self.linear_part(features.transpose(2, 1)).reshape(FM.size(0), -1)  # [batch_size, embed_size]
        FM = linear_part + FM  # [batch_size, embed_size]

        # deepu部分的代码+
        # 第一种deep方案
        deep = self.mlp_layers(features.reshape(FM.size(0), -1))
        # #第二种deep方案 [self.maxlen,self.maxlen,self.maxlen,1] 
        # deep = self.mlp_layers(features.transpose(2,1)).reshape(FM.size(0),-1)

        output = self.sigmoid(FM + deep)

        return output

    def forward(self, batch):

        # 数据准备

        item_seq, labels = batch

        # 2 获得有画后的特征  [b,es]

        x = self.log2feats(item_seq)

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

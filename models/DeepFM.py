import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel
# from utils import mlp, cos, euclid


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

    def __init__(self, layers, dropout):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = nn.ReLU()
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class DeepFM(BaseModel):
    def __init__(self, args, dataset, device, max_len):
        super(DeepFM, self).__init__(dataset, device, max_len)
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        """

        self.num_factors = args.hidden_unit  # embeding size是多大

        self.layers = args.dfm_layers  # mlp每层的参数
        self.drop_prob = args.drop_prob  # FM mlp的dropout分别是什么

        self.loss_type = args.loss_type

        self.embeddings = nn.Embedding(self.n_item + 1, self.num_factors)
        self.biases = nn.Embedding(self.n_item + 1, 1)  # 每个item的
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        FM_modules.append(nn.BatchNorm1d(self.num_factors))
        FM_modules.append(nn.Dropout(self.drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)
        # FM的Linear-part
        # [b,maxlen,es]-转置->[b,es,maxlen] -linear->[b,es, 1 ]-转置->[b, 1,es] es = embedding size 
        self.linear_part = nn.Linear(self.max_len, 1)

        # for deep layers
        in_dim = self.num_factors * self.max_len
        self.layers = [in_dim] + self.layers
        self.mlp_layers = MLPLayers(self.layers, self.drop_prob[-1])

        self.sigmoid = nn.Sigmoid()

        # Output
        self.output = nn.Linear(self.num_factors, self.n_item + 1)
        
        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        else:
            logging.critical(f"{self.loss_type} has not been implemented yet.")
            raise NotImplementedError

    @classmethod
    def code(cls):
        return 'deepfm'

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

        item_seq = batch[0]

        # 2 获得有画后的特征  [b,es]

        x = self.log2feats(item_seq)

        pred = F.linear(x, self.embeddings.weight)

        return pred  # B * L * D --> B * L * N

    @torch.no_grad()
    def predict(self, batch):
        candidates = batch[1]

        scores = self.forward(batch)

        scores = scores.gather(1, candidates)

        return scores
    
    @torch.no_grad()
    def full_sort_predict(self, batch):
        scores = self.forward(batch)
        return scores
    
    def calculate_loss(self, batch):
        labels = batch[1]

        cl = labels.squeeze()
        logits = self.forward(batch)
        loss = self.loss_fct(logits, cl)

        return logits, loss

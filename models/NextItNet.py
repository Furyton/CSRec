# -*- coding: utf-8 -*-
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
NextItNet
################################################
Reference:
    Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation" in WSDM 2019.
Reference code:
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch
"""
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_

from models.base import BaseModel
from models.loss import RegLoss

# from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.loss import RegLoss, BPRLoss


class NextItNet(BaseModel):
    r"""The network architecture of the NextItNet model is formed of a stack of holed convolutional layers, which can
    efficiently increase the receptive fields without relying on the pooling operation.
    Also residual block structure is used to ease the optimization for much deeper networks.
    Note:
        As paper said, for comparison purpose, we only predict the next one item in our evaluation,
        and then stop the generating process. Although the number of parameters in residual block (a) is less
        than it in residual block (b), the performance of b is better than a.
        So in our model, we use residual block (b).
        In addition, when dilations is not equal to 1, the training may be slow. To  speed up the efficiency, please set the parameters "reproducibility" False.
    """

    def __init__(self, args, dataset, device, max_len, temp: float=1.0):
        super(NextItNet, self).__init__(dataset, device, max_len, temp)

        # load parameters info
        self.embedding_size = args.embedding_size

        self.residual_channels = self.embedding_size

        self.block_num = args.block_num

        self.dilations = args.dilations * self.block_num

        self.kernel_size = args.kernel_size

        self.reg_weight = args.reg_weight
        self.loss_type = args.loss_type

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_item + 1, self.embedding_size, padding_idx=0)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.embedding_size)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        else:
            logging.critical(f"{self.loss_type} has not been implemented yet.")
            raise NotImplementedError

        self.reg_loss = RegLoss()

        # parameters initialization
        self.apply(self._init_weights)

    @classmethod
    def code(cls):
        return 'nextitnet'

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.n_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, batch):
        item_seq = batch[0]

        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        seq_output = self.final_layer(hidden)  # [batch_size, embedding_size]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return logits / self.temperature

    def reg_loss_rb(self):
        r"""
        L2 loss on residual blocks
        """
        loss_rb = 0
        if self.reg_weight > 0.0:
            for name, parm in self.residual_blocks.named_parameters():
                if name.endswith('weight'):
                    loss_rb += torch.norm(parm, 2)
        return self.reg_weight * loss_rb
    
    def calculate_loss(self, batch):
        labels = batch[1]
        cl = labels.squeeze()
        logits = self.forward(batch)
        loss = self.loss_fct(logits, cl)

        reg_loss = self.reg_loss([self.item_embedding.weight, self.final_layer.weight])

        # loss = loss + self.reg_weight * reg_loss + self.reg_loss_rb()

        return logits, loss, self.reg_weight * reg_loss + self.reg_loss_rb()

    def predict(self, batch):
        candidates = batch[1]

        scores = self.forward(batch)
        scores = scores.gather(1, candidates)

        return scores

    def full_sort_predict(self, batch):
        scores = self.forward(batch)

        return scores


class ResidualBlock_a(nn.Module):
    r"""
    Residual block (a) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_a, self).__init__()

        half_channel = out_channel // 2
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv1 = nn.Conv2d(in_channel, half_channel, kernel_size=(1, 1), padding=0)

        self.ln2 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(half_channel, half_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)

        self.ln3 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv3 = nn.Conv2d(half_channel, out_channel, kernel_size=(1, 1), padding=0)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]

        out = F.relu(self.ln1(x))
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)

        out2 = F.relu(self.ln2(out))
        out2 = self.conv_pad(out2, self.dilation)
        out2 = self.conv2(out2).squeeze(2).permute(0, 2, 1)

        out3 = F.relu(self.ln3(out2))
        out3 = out3.permute(0, 2, 1).unsqueeze(2)
        out3 = self.conv3(out3).squeeze(2).permute(0, 2, 1)
        return out3 + x

    def conv_pad(self, x, dilation):  # x: [batch_size, seq_len, embed_size]
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        # padding operation  args：(left,right,top,bottom)
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


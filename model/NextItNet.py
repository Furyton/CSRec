r"""
NextItNet based on RecBole's implementation
################################################

Reference:
    Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation" in WSDM 2019.

Reference code:
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch

"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_
from utils import mlp, cos, euclid


class NextItNet(nn.Module):
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

    def __init__(self, args):
        super(NextItNet, self).__init__()

        # load parameters info
        self.device = args.device

        self.num_item = args.num_item + 1
        self.embedding_size = args.d_model
        self.residual_channels = args.d_model
        self.block_num = args.block_num
        self.dilations = args.dilations * self.block_num
        self.kernel_size = args.kernel_size
        self.enabel_res_parameter = args.enable_res_parameter
        self.embedding_sharing = args.embedding_sharing
        self.enable_sample = args.enable_sample

        # define layers and loss
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_size, padding_idx=0)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation,
                enable_res_parameter=self.enabel_res_parameter
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.le_share=args.le_share
        if args.le_share== 0 or args.soft_target== 'unshare':
            self.output = nn.Linear(self.residual_channels, self.num_item) # self.num_item已经加1了
        
        self.soft_taget = args.soft_target
        if args.soft_target== 'mlp':
            self.mlp = mlp(args.mlp_hiddent, args.dropout, self.num_item)


        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)


    def log2feats(self,x):
        #1 mask
        #1 获取特征
        item_seq_emb = self.item_embedding(x)  # [batch_size, seq_len, embed_size]
        

        #3 Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb) # [batch_size, seq_len, embed_size]
        
        return dilate_outputs


    def forward(self, batch):
        
        item_seq, labels = batch



        # #1 获取特征
        # item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        

        # #3 Residual locks
        # dilate_outputs = self.residual_blocks(item_seq_emb) # [batch_size, seq_len, embed_size]
        x = self.log2feats(item_seq)

        # 返回的参数中第一个是sahre 第二个是unshare

        if self.training: #使用这种方式减少现存的占用 
            x = x[labels>0]
        
        
        if self.le_share:
            pred = F.linear(x, self.item_embedding.weight)
        else:
            pred = self.output(x)

        # 在le loss中使用的soft target
        if self.soft_taget == 'mlp':
            soft_target = self.mlp(x)
        elif self.soft_taget == 'cos':
            soft_target = cos(x,self.item_embedding.weight)
        elif self.soft_taget == 'euclid':
            x = x.view(-1,x.size(-1))
            soft_target = euclid(x,self.item_embedding.weight)
            if self.training== False:
                soft_target = soft_target.view(labels.size(0),-1,soft_target.size(-1))
        elif self.soft_taget == 'unshare':
            soft_target = self.output(x)
        elif self.soft_taget == 'share':
            soft_target = F.linear(x, self.item_embedding.weight)
            

        return pred, soft_target  # [batch_size, seq_len, num_item] 


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


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, enable_res_parameter=False):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

        self.enable = enable_res_parameter
        self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))

        if self.enable:
            return self.a * out2 + x
        else:
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

from numpy import mod
from dataset import TrainDataset, EvalDataset
from process import Trainer
from args import args
import pandas as pd
import torch
import torch.utils.data as Data
from model.BERT import BERT
from model.NextItNet import NextItNet
from model.nfm import NFM
from model.deepfm import DeepFM
from tqdm import tqdm
from visdom import Visdom
from sklearn import manifold, datasets

# t-SNE 降维
def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
    tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result

def Visualization(vis, result, labels,title):                                 
    # vis: Visdom对象                                                           
    # result: 待显示的数据，这里为t_SNE()函数的输出                                          
    # label: 待显示数据的标签                                                         
    # title: 标题                                                               
    vis.scatter(                                                              
        X = result,                                                          
        Y = labels+1,           # 将label的最小值从0变为1，显示时label不可为0                
       opts=dict(markersize=3,title=title),                                   
    ) 


model_path = '/data/wyzhang/experiment/le4-rec-baseline/Model-nfmle_share-1le_res-0.1D-64_Lr-0.001_Loss-ce/model.pkl'

#1 获得数据 
args.test_batch_size = 100

test_dataset = EvalDataset(args.max_len, args.sample_size, mode='test', enable_sample=args.enable_sample,
                            path=args.data_path, model=args.model, device=args.device)
test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)


#2 加载模型
args.model = 'nfm'
if args.model == 'bert' or args.model == 'sasrec' :
    model = BERT(args)
elif args.model == 'nextitnet':
    model = NextItNet(args)
elif args.model == 'nfm':
    model = NFM(args)
elif args.model == 'deepfm':
    model = DeepFM(args)
else:
    raise NotImplementedError


model.load_state_dict(torch.load('params.pkl'))
model.eval()
model.to(torch.device(args.device))
#3 获取用户embedding
User = []
tqdm_data_loader = tqdm(test_loader)

for idx, batch in enumerate(tqdm_data_loader):
    batch = [x.to(args.device) for x in batch]
    seq, _ = batch
    tmp = model.log2feats(seq)
    User.append(tmp.cpu())
    if (idx+1) %10==0:
        break
data = torch.cat(User, dim=1)
#4 降维
data = t_SNE(data,2)
#5 画图
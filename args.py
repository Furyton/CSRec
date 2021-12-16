import argparse
import time
import os
import json
import pandas as pd

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--data_path', type=str, default="/data/wushiguang-slurm/code/soft-rec/tiny_ml.csv")
parser.add_argument('--save_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--mask_prob', type=float, default=0.3)
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)

# model args
parser.add_argument('--model', type=str, default='sasrec', choices=['bert', 'sasrec', 'nextitnet', 'nfm', 'deepfm', 'GRU4Rec'])
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eval_per_steps', type=int, default=1)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--neg_samples', type=int, default=100)
parser.add_argument('--samples_ratio', type=float, default=0.1)

parser.add_argument('--sample_strategy', type=str, default='raw_random', choices=['seq', 'batch', 'mask', 'raw_random'])

# bert args
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--d_ffn', type=int, default=512)
parser.add_argument('--bert_layers', type=int, default=10)
parser.add_argument('--embedding_sharing', type=int, default=0)

# nextitnet args
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--block_num', type=int, default=10)
parser.add_argument('--dilations', type=list, default=[1,4])

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=0.95)
parser.add_argument('--lr_decay_steps', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=15)
parser.add_argument('--metric_ks', type=list, default=[10])
parser.add_argument('--best_metric', type=str, default='NDCG@10')

# FM超参
parser.add_argument('--nfm_layers', type=list, default=[128,64]) #  第二个值要和d_model 一致
parser.add_argument('--dfm_layers', type=list, default=[512,64]) # 第二个值要和d_model 一致
parser.add_argument('--drop_prob', type=list, default=[0.1,0.1]) # 第一个是FM的超参，第二个是mlp的超can
parser.add_argument('--act_function', default='relu', type=str , help='mlp模型中使用的activate function relu sigmoid tanh')

# gru超参

parser.add_argument('--num_layers', type=int, default=3) #  第二个值要和d_model 一致
parser.add_argument('--hidden_size', type=int, default=128) #  第二个值要和d_model 一致


#LE4Rec

parser.add_argument('--le_share', type=int, default=0) # 1表示在kl中使用share作为prediction   0表示使用unshare
parser.add_argument('--le_res', type=float, default=0.1) # 1表示在kl中使用share作为prediction   0表示使用unshare
parser.add_argument('--le_t', type=float, default=2) # 1表示在kl中使用share作为prediction   0表示使用unshare
parser.add_argument('--enable_sample', type=int, default=1) # 1使用采样版本 0full的版本 与loss一起选择相对应
parser.add_argument('--loss_type', type=str, default='le', choices=['ce', 'bce', 'bpr', 'le'])
parser.add_argument('--mlp_hiddent', type=list, default=[64,1024]) # 第一个值和d_model一致
parser.add_argument('--soft_taget', type=str, default='euclid', choices=['share', 'unshare', 'mlp', 'cos','euclid'])

args = parser.parse_args()
# other args

DATA = pd.read_csv(args.data_path, header=None)
num_item = DATA.max().max()
del DATA
args.num_item = int(num_item)

if args.save_path == 'None':
    loss_str = args.loss_type
    path_str = 'Model-' + args.model +'_le_share-' + str(args.le_share)+'-le_res-' + str(args.le_res) + '-le_t-' + str(args.le_t) + \
               '_Lr-' + str(args.lr) + '_Loss-' + loss_str + '_sample-' + str(args.enable_sample)+ '_soft_taget-' + str(args.soft_taget)
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()

from dataset import TrainDataset, EvalDataset
from process import Trainer
from args import args
import pandas as pd

import torch.utils.data as Data
from model.BERT import BERT
from model.NextItNet import NextItNet
from model.nfm import NFM
from model.deepfm import DeepFM
from model.GRU4rec import GRU4Rec


def main():
    train_dataset = TrainDataset(args.mask_prob, args.max_len, args.data_path, model=args.model, device=args.device)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # val_dataset = EvalDataset(args.max_len, args.sample_size, mode='val', enable_sample=args.enable_sample,
    #                           path=args.data_path, model=args.model, device=args.device)
    # val_loader = Data.DataLoader(val_dataset, batch_size=args.val_batch_size)

    test_dataset = EvalDataset(args.max_len, args.sample_size, mode='test', enable_sample=args.enable_sample,
                               path=args.data_path, model=args.model, device=args.device)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initialize ends')

    if args.model == 'bert' or args.model == 'sasrec':
        model = BERT(args)
    elif args.model == 'nextitnet':
        model = NextItNet(args)
    elif args.model == 'nfm':
        model = NFM(args)
    elif args.model == 'deepfm':
        model = DeepFM(args)
    elif args.model == 'GRU4Rec':
        model = GRU4Rec(args)
    else:
        raise NotImplementedError

    trainer = Trainer(args, model, train_loader, test_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    main()

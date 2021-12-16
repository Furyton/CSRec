import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data


# use 0 to padding


class TrainDataset(Data.Dataset):
    def __init__(self, mask_prob, max_len, path, model, device):
        self.data = pd.read_csv(path).replace(-1, 0).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.model = model
        self.device = device

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        if self.model == 'bert':
            seq = self.data[index, :-2]

            tokens = []
            labels = []

            for s in seq:
                if s != 0:
                    prob = random.random()
                    if prob < self.mask_prob:
                        tokens.append(self.mask_token)
                        labels.append(s)
                    else:
                        tokens.append(s)
                        labels.append(0)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]
            mask_len = self.max_len - len(tokens)

            seq = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

        elif self.model == 'sasrec' or self.model == 'nextitnet' or self.model == 'GRU4Rec':
            seq = self.data[index, -self.max_len - 3:-3].tolist()
            pos = self.data[index, -self.max_len - 2:-2].tolist()
            pos = pos[-len(seq):]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
            padding_len = self.max_len - len(pos)
            labels = [0] * padding_len + pos

        else:  # NFM DeepFM
            seq = self.data[index, -self.max_len - 3:-3].tolist()
            labels = [self.data[index, -2].tolist()]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
            # padding_len = self.max_len - len(pos)
            # pos = [0] * padding_len + pos               

        return torch.LongTensor(seq), torch.LongTensor(labels)


class EvalDataset(Data.Dataset):
    def __init__(self, max_len, sample_size, mode, enable_sample, path, model, device):
        self.data = pd.read_csv(path).replace(-1, 0).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.max_len = max_len
        self.sample_size = sample_size
        self.mode = mode
        self.enable_sample = enable_sample
        self.model = model
        self.device = device

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2] if self.mode == 'val' else self.data[index, :-1]
        pos = self.data[index, -2] if self.mode == 'val' else self.data[index, -1]
        negs = []

        # if self.enable_sample:
        #     seen = set(seq)
        #     seen.update([pos])
        #     while len(negs) < self.sample_size:
        #         candidate = np.random.randint(0, self.num_item) + 1
        #         while candidate in seen or candidate in negs:
        #             candidate = np.random.randint(0, self.num_item) + 1
        #         negs.append(candidate)

        #     answers = [pos] + negs
        #     labels = [1] + [0] * len(negs)

        #     seq = list(seq)
        #     if self.model == 'bert':
        #         seq = seq + [self.mask_token]

        #     seq = seq[-self.max_len:]
        #     padding_len = self.max_len - len(seq)
        #     seq = [0] * padding_len + seq

        #     return torch.LongTensor(seq).to(self.device), torch.LongTensor(answers).to(self.device), torch.LongTensor(
        #         labels).to(self.device)

        # else:
        seq = list(seq)
        if self.model == 'bert':
            seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        answers = [pos]
        return torch.LongTensor(seq).to(self.device), torch.LongTensor(answers).to(self.device)

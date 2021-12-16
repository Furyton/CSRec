import math
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed


def neg_sample(seq, labels, num_item, sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs


class LE:
    def __init__(self, model, args):
        self.model = model
        self.le_share = args.le_share
        self.b = args.le_res
        self.t = args.le_t
        self.enable_sample = args.enable_sample
        self.num_item = args.num_item
        self.sample_ratio = args.samples_ratio
        if self.enable_sample:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.device = args.device

    def compute(self, batch):

        seqs, labels = batch

        pred, soft_target = self.model(batch)  # B * L * N
        cl = labels[labels > 0]

        if self.enable_sample:
            # 负样本采样 
            negs = neg_sample(seqs, labels, self.num_item, int(self.num_item * self.sample_ratio))
            negs = torch.LongTensor(list(negs)).repeat(len(cl), 1).to(self.device)
            target = torch.cat((cl.unsqueeze(1), negs), 1)
            # 采样后的one_hot
            one_hot = [1] + [0] * negs.size(-1)
            one_hot = torch.LongTensor(one_hot).repeat(negs.size(0), 1).to(torch.device(self.device))
            # 抽取采样后的结果
            pred = pred.gather(dim=1, index=target)
            soft_target = soft_target.gather(dim=1, index=target)

            # 标签
            label = torch.LongTensor([0] * pred.size(0)).to(torch.device(self.device))
            # 计算kl的值
            tmp = F.kl_div((pred.softmax(dim=-1) / self.t).log(),
                           ((soft_target.softmax(dim=-1) / self.t) + one_hot).softmax(dim=-1), reduction='batchmean')
            if ~torch.isinf(tmp):
                loss = (1 - self.b) * self.ce(pred, label)
                loss += pow(self.t, 2) * self.b * tmp
            else:
                loss = self.ce(pred, label)

        else:
            # 标签转换成 one_hot
            cl_onehot = torch.nn.functional.one_hot(cl, num_classes=soft_target.size(-1))
            tmp = F.kl_div((pred.softmax(dim=-1) / self.t).log(),
                           ((soft_target.softmax(dim=-1) / self.t) + cl_onehot).softmax(dim=-1), reduction='batchmean')
            if ~torch.isinf(tmp):
                loss = (1 - self.b) * self.ce(pred, cl)
                loss += pow(self.t, 2) * self.b * tmp
            else:
                loss = self.ce(pred, cl)

        return loss


class CE:
    def __init__(self, model, args):
        self.model = model
        self.le_share = args.le_share
        self.enable_sample = args.enable_sample
        self.num_item = args.num_item
        self.sample_ratio = args.samples_ratio
        self.device = args.device

        if self.enable_sample:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):

        seqs, labels = batch

        pred, _ = self.model(batch)  # B * L * N
        cl = labels[labels > 0]
        if self.enable_sample:
            negs = neg_sample(seqs, labels, self.num_item, int(self.num_item * self.sample_ratio))
            negs = torch.LongTensor(list(negs)).repeat(len(cl), 1).to(self.device)
            target = torch.cat((cl.unsqueeze(1), negs), 1)  # 从全部的数据中提取sample的数据

            pred = pred.gather(dim=1, index=target)
            label = torch.LongTensor([0] * pred.size(0)).to(torch.device(self.device))  # lable都放在了第一维即index=0
            loss = self.ce(pred, label)
        else:
            loss = self.ce(pred, cl)

        return loss


class BCE:
    def __init__(self, model, neg_samples, num_item, device, sample_strategy):
        self.model = model
        self.neg_samples = neg_samples
        self.num_item = num_item
        self.device = device
        self.sample_strategy = sample_strategy
        self.bce = nn.BCEWithLogitsLoss()

    def generate_item_neg_items_random(self, seq, mask):
        neg_items = []
        items = np.arange(0, self.num_item + 1)
        for i in range(len(seq)):
            _mask = mask[i]
            _mask = (_mask[_mask > 0]).tolist()
            seen = list(set(seq[i].tolist()))
            if self.num_item + 1 in seen:
                seen.remove(self.num_item + 1)
            _items = np.delete(items, seen)
            for j in range(len(_mask)):
                neg_items.append([_mask[j]] + _items[np.random.randint(0, len(_items), self.neg_samples)].tolist())
        return torch.LongTensor(neg_items)

    def generate_neg_items(self, seq, mask):
        neg_items = []
        for i in range(seq.shape[0]):
            neg_items += self.generate_neg_items_for_user(seq[i], mask[i])
        return torch.LongTensor(neg_items)

    def generate_neg_items_for_user(self, seq, mask):
        mask = mask[mask > 0]
        neg_items = []
        candidate = []
        for i in range(self.neg_samples):
            _item = np.random.randint(1, self.num_item + 1)
            while _item in seq or _item in mask:
                _item = np.random.randint(1, self.num_item + 1)
            candidate.append(_item)
        for i in range(len(mask)):
            neg = [mask[i]] + candidate
            neg_items.append(neg)
        return neg_items

    def generate_neg_items_batch(self, seq, mask):
        mask = mask[mask > 0]
        seq = list(set(seq.view(-1).tolist()))
        neg_items = []
        candidate = []
        for i in range(self.neg_samples):
            _item = np.random.randint(1, self.num_item + 1)
            while _item in seq or _item in mask:
                _item = np.random.randint(1, self.num_item + 1)
            candidate.append(_item)
        for i in range(mask.shape[0]):
            neg = [mask[i]] + candidate
            neg_items.append(neg)
        return torch.LongTensor(neg_items)

    def generate_neg_items_using_mask(self, seq, mask):
        mask = (mask[mask > 0]).tolist()
        mask_set = set(mask)
        self.neg_samples = len(mask_set) - 1  #
        neg_items = []
        for i in range(len(mask)):
            tmp = copy.deepcopy(mask_set)
            tmp.remove(mask[i])
            neg_items.append([mask[i]] + list(tmp))
        return torch.LongTensor(neg_items)

    def compute(self, batch):
        seqs, mask = batch

        if self.sample_strategy == 'batch':
            neg_items = self.generate_neg_items_batch(seqs, mask).to(self.device)
        elif self.sample_strategy == 'seq':
            neg_items = self.generate_neg_items(seqs, mask).to(self.device)
        elif self.sample_strategy == 'mask':
            neg_items = self.generate_neg_items_using_mask(seqs, mask).to(self.device)
        elif self.sample_strategy == 'raw_random':
            neg_items = self.generate_item_neg_items_random(seqs, mask).to(self.device)
        else:
            raise NotImplementedError

        outputs = self.model(seqs)  # B * L * N
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*L) * N
        mask = mask.view(-1)

        preds = outputs[mask > 0]  # num_of_mask * N
        values = preds.gather(1, neg_items)
        labels = torch.Tensor([[1] + [0] * self.neg_samples for i in range(values.shape[0])]).to(self.device)

        loss = self.bce(values, labels)
        return loss


class BPR:
    def __init__(self, model, neg_samples, num_item, device, sample_strategy):
        self.model = model
        self.neg_samples = neg_samples
        self.num_item = num_item
        self.device = device
        self.sample_strategy = sample_strategy
        self.bpr = self.bpr_loss

    def generate_item_pairs_raw_random(self, seq, mask):
        item_pairs = []
        items = np.arange(0, self.num_item + 1)
        for i in range(len(seq)):
            _mask = mask[i]
            _mask = (_mask[_mask > 0]).tolist()
            seen = list(set(seq[i].tolist()))
            if self.num_item + 1 in seen:
                seen.remove(self.num_item + 1)
            _items = np.delete(items, seen)
            for j in range(len(_mask)):
                item_pairs.append([_mask[j]] + _items[np.random.randint(0, len(_items), self.neg_samples)].tolist())
        return torch.LongTensor(item_pairs)

    def generate_item_pairs_parallel(self, seqs, masks):
        n_processor = 4
        size = seqs.shape[0] // n_processor
        result = Parallel(n_jobs=n_processor, verbose=1)(
            delayed(self.generate_item_pairs)(seqs[i * size:(i + 1) * size], masks[i * size:(i + 1) * size]) for i in
            range(n_processor))
        return result

    def generate_item_pairs(self, seq, mask):
        item_pairs = []
        for i in range(seq.shape[0]):
            item_pairs += self.generate_item_pairs_for_user(seq[i], mask[i])
        return torch.LongTensor(item_pairs)

    def generate_item_pairs_for_user(self, seq, mask):
        mask = mask[mask > 0]
        item_pairs = []
        candidate = []
        for i in range(self.neg_samples):
            _item = np.random.randint(1, self.num_item + 1)
            while _item in seq or _item in mask:
                _item = np.random.randint(1, self.num_item + 1)
            candidate.append(_item)
        for i in range(len(mask)):
            pair = [mask[i]] + candidate
            item_pairs.append(pair)
        return item_pairs

    def generate_item_pairs_batch(self, seq, mask):
        mask = mask[mask > 0]
        seq = list(set(seq.view(-1).tolist()))
        item_pairs = []
        candidate = []
        for i in range(self.neg_samples):
            _item = np.random.randint(1, self.num_item + 1)
            while _item in seq or _item in mask:
                _item = np.random.randint(1, self.num_item + 1)
            candidate.append(_item)
        for i in range(mask.shape[0]):
            pair = [mask[i]] + candidate
            item_pairs.append(pair)
        return torch.LongTensor(item_pairs)

    def generate_item_pairs_using_mask(self, seq, mask):
        mask = (mask[mask > 0]).tolist()
        mask_set = set(mask)
        self.neg_samples = len(mask_set) - 1
        item_pairs = []
        for i in range(len(mask)):
            tmp = copy.deepcopy(mask_set)
            tmp.remove(mask[i])
            item_pairs.append([mask[i]] + list(tmp))
        return torch.LongTensor(item_pairs)

    def compute(self, batch):
        seqs, mask = batch
        """
        item_pairs = []
        result = self.generate_item_pairs_parallel(seqs, mask)  # (num_of_neg*num_of_mask) * 2
        for _ in result:
            item_pairs += _
        item_pairs = torch.LongTensor(item_pairs).to(self.device)
        """

        if self.sample_strategy == 'batch':
            item_pairs = self.generate_item_pairs_batch(seqs, mask).to(self.device)
        elif self.sample_strategy == 'seq':
            item_pairs = self.generate_item_pairs(seqs, mask).to(self.device)
        elif self.sample_strategy == 'mask':
            item_pairs = self.generate_item_pairs_using_mask(seqs, mask).to(self.device)
        elif self.sample_strategy == 'raw_random':
            item_pairs = self.generate_item_pairs_raw_random(seqs, mask).to(self.device)
        else:
            raise NotImplementedError

        outputs = self.model(seqs)  # B * L * N
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*L) * N
        mask = mask.view(-1)

        preds = outputs[mask > 0]  # num_of_mask * N
        scores = preds.gather(1, item_pairs)
        loss = self.bpr(scores[:, 1:].reshape(-1),
                        scores[:, 0].repeat(self.neg_samples, 1).transpose(0, 1).contiguous().view(-1))

        return loss

    @staticmethod
    def bpr_loss(neg_scores, pos_scores):
        return torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))


class MutualInformationLoss(nn.Module):
    def __init__(self, measure):
        super(MutualInformationLoss, self).__init__()
        self.measure = measure
        self.cr = nn.CrossEntropyLoss()
        print('CL measure method:' + measure)

    def forward(self, local_rep, global_rep, mask):
        #  local_rep in shape B * L * nD
        #  global_rep in shape B * nD
        #  mask in shape B * L
        local_rep = local_rep[mask]  # B * L * nD -> num of none mask * nD
        score = torch.mm(local_rep, global_rep.t())  # num of none mask * B , batch as label
        if self.measure == 'InfoNCE':
            label = self.create_label(mask)
            loss = self.cr(score, label)
        else:
            seq_lens = torch.sum(mask, dim=1)
            pos_mask, neg_mask = self.create_masks(seq_lens)
            num_nodes = pos_mask.size(0)
            num_graphs = pos_mask.size(1)
            E_pos = self.get_positive_expectation(score * pos_mask, self.measure, average=False).sum()
            E_pos = E_pos / num_nodes
            E_neg = self.get_negative_expectation(score * neg_mask, self.measure, average=False).sum()
            E_neg = E_neg / (num_nodes * (num_graphs - 1))
            loss = E_neg - E_pos

        return loss

    @staticmethod
    def create_label(mask):
        bsz = mask.shape[0]
        label = torch.arange(bsz).unsqueeze(-1).expand(mask.size())
        label = label[mask].to(mask.device).long()
        return label

    @staticmethod
    def create_masks(seq_lens):
        pos_mask = torch.zeros(torch.sum(seq_lens), len(seq_lens))
        neg_mask = torch.ones(torch.sum(seq_lens), len(seq_lens))
        temp = 0
        for idx in range(len(seq_lens)):
            for j in range(temp, seq_lens[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += seq_lens[idx]

        return pos_mask.to(seq_lens.device), neg_mask.to(seq_lens.device)

    def get_positive_expectation(self, p_samples, measure, average=True):
        """Computes the positive part of a divergence / difference.
        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(- p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise NotImplementedError

        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, measure, average=True):
        """Computes the negative part of a divergence / difference.
        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = self.log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            raise NotImplementedError

        if average:
            return Eq.mean()
        else:
            return Eq

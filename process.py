import time
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from loss import CE, BCE, BPR, LE
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    def __init__(self, args, model, train_loader, test_loader):
        #  self.result_file = open(self.save_path + '/result.txt', 'a+')

        self.args = args
        self.device = args.device
        print(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.loss = args.loss_type
        self.neg_samples = args.neg_samples
        self.enable_sample = args.enable_sample
        

        self.model = model.to(torch.device(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=True)

        
        if self.loss == 'ce':
            self.cr = CE(self.model, args)
        elif self.loss == 'bce':
            self.cr = BCE(self.model, self.neg_samples, self.args.num_item, self.device, self.args.sample_strategy)
        elif self.loss == 'bpr':
            self.cr = BPR(self.model, self.neg_samples, self.args.num_item, self.device, self.args.sample_strategy)
        elif self.loss == 'le':
            self.cr = LE(self.model, args)

        self.num_epoch = args.num_epoch
        self.metric_ks = args.metric_ks
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        self.result_file = open(self.save_path + '/result.txt', 'w')

        self.step = 0
        self.metric = args.best_metric
        self.best_metric = -1e9
        # le
        self.le_share = args.le_share

    def train(self):
        # BERT training
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            print('train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
                  
        print(self.eval_model('test'), file=self.result_file)
        self.result_file.close()

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                self.sample_time = 0
                metric = {}
                for mode in ['test']:
                    metric[mode] = self.eval_model(mode)
                print(metric)
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                if metric['test'][self.metric] > self.best_metric:
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric['test'][self.metric]
                self.model.train()

        return loss_sum / idx, time.perf_counter() - t0

    def eval_model(self, mode):
        self.model.eval()
        tqdm_data_loader = tqdm(self.val_loader) if mode == 'val' else tqdm(self.test_loader)
        metrics = {}

        with torch.no_grad():
            for idx, eval_batch in enumerate(tqdm_data_loader):
                eval_batch = [x.to(self.device) for x in eval_batch]
                metrics_batch = self.compute_metrics(eval_batch)

                for k, v in metrics_batch.items():
                    if not metrics.__contains__(k):
                        metrics[k] = v
                    else:
                        metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v / idx
        return metrics

    def compute_metrics(self, batch):
        # if self.enable_sample:
        #     seqs, answers, labels = batch
        #     scores = self.model(seqs)
        #     scores = scores[:, -1, :]  # the prediction score of the last `mask` token in seq, B * N
        #     scores = scores.gather(1, answers)  # only consider positive and negative items' score

        # else:
        seqs, answers = batch
        scores = self.model(batch)
        if self.args.model == 'nfm' or self.args.model == 'deepfm':
            scores = scores[0]

        else:  # bert sasrec nextit
            scores = scores[0][:, -1, :]

        labels = torch.zeros(seqs.shape[0], self.args.num_item + 1).to(self.device)
        row = []
        col = []
        seqs = seqs.tolist()
        answers = answers.tolist()
        for i in range(len(answers)):
            seq = list(set(seqs[i] + answers[i]))
            seq.remove(answers[i][0])
            if self.args.num_item + 1 in seq:
                seq.remove(self.args.num_item + 1)
            row += [i] * len(seq)
            col += seq
            labels[i][answers[i]] = 1
        scores[row, col] = -1e9
        metrics = self.recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    @staticmethod
    def recalls_and_ndcgs_for_ks(scores, labels, ks):  # MRR ，HR和Recall的区别
        metrics = {}

        answer_count = labels.sum(1)

        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                         labels.sum(1).float())).mean().cpu().item()

            position = torch.arange(2, 2 + k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()

        return metrics

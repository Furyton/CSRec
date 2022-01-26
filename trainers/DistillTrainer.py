import logging

import torch
import torch.optim as optm
import torch.utils.data as data_utils
from configuration.config import *
from loggers import LoggerService
from models.base import BaseModel

from trainers.loss import SoftLoss
from trainers.BasicTrainer import Trainer
from trainers.utils import assert_model_device, recalls_ndcgs_and_mrr_for_ks
from utils import AverageMeterSet


class DistillTrainer(Trainer):
    def __init__(self,
                 args,
                 optim: optm.Optimizer,
                 lr_sched: optm.lr_scheduler,
                 train_loader: data_utils.DataLoader,
                 val_loader: data_utils.DataLoader,
                 test_loader: data_utils.DataLoader,
                 model_list: list,
                 tag_list: list,
                 logger: LoggerService,
                 device: str,
                 accum_iter: int = 0):
        super(Trainer, self).__init__(args, device)

        self.optimizer = optim
        self.lr_scheduler = lr_sched
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model_list = model_list # 0: main model, 1: teacher
        self.tag_list = tag_list # 0: main model, 1: teacher
        self.tag = self.tag_list[0]
        self.logger = logger
        self.accum_iter = accum_iter

        self.epoch = 0

        self.model = self.model_list[0]
        self.auxiliary_model = self.model_list[1]

        for model, tag in zip(self.model_list, self.tag_list):
            assert_model_device(model, self.device, tag, args.device_idx)

        self.loss_fct = SoftLoss(self.auxiliary_model, args)

        self.iter_per_epoch = len(self.train_loader) * self.batch_size
        self.tot_iter = self.num_epochs * self.iter_per_epoch

        logging.info('{} iter per epoch'.format(self.iter_per_epoch))

        logging.info(f'Test mentor model: {self.tag_list[1]}')
        logging.info(f"result:\n{self.test_mentor()}")

    def train(self):
        super().train()

        self.loss_fct.debug_summary()

    def _calculate_metrics(self, model: BaseModel, batch):
        batch = [x.to(self.device) for x in batch]

        if self.enable_neg_sample:
            logging.fatal("codes for evaluating with negative candidates has bug")
            raise NotImplementedError(
                "codes for evaluating with negative candidates has bug")
            scores = model.predict(batch)
        else:
            # seqs, answer, ratings, ... = batch
            seqs = batch[0]
            answer = batch[1]
            ratings = batch[2]

            batch_size = len(seqs)
            labels = torch.zeros(
                batch_size, self.num_items + 1, device=self.device)
            scores = model.full_sort_predict(batch)

            row = []
            col = []

            for i in range(batch_size):
                seq = list(set(seqs[i].tolist()) | set(answer[i].tolist()))
                seq.remove(answer[i][0].item())
                if self.num_items + 1 in seq:
                    seq.remove(self.num_items + 1)
                row += [i] * len(seq)
                col += seq
                labels[i][answer[i]] = 1
            scores[row, col] = -1e9

        metrics = recalls_ndcgs_and_mrr_for_ks(
            scores, labels, self.metric_ks, ratings)
        return metrics

    def test_mentor(self):
        self.auxiliary_model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.test_loader

            for batch_idx, batch in enumerate(iterator):

                metrics = self._calculate_metrics(self.auxiliary_model, batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

        average_metrics = average_meter_set.averages()
        logging.info(average_metrics)

        return average_metrics   

    @classmethod
    def code(cls):
        return 'distill'

    def _create_log_data(self, metrics: dict = None):
        data_dict = {
            STATE_DICT_KEY: self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
            EPOCH_DICT_KEY: self.epoch,
            ACCUM_ITER_DICT_KEY: self.accum_iter
        }

        if metrics is not None:
            data_dict.update(metrics)

        return data_dict

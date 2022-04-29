import json
import logging
from pathlib import Path
import time

import torch
import torch.optim as optm
import torch.utils.data as data_utils
from configuration.config import *
from models.base import BaseModel
from scheduler.utils import get_best_state_path, load_state_from_given_path, load_state_from_local

from trainers.BaseTrainer import AbstractBaseTrainer
from loggers import LoggerService
from trainers.loss import BasicLoss 
from trainers.utils import assert_model_device, recalls_ndcgs_and_mrr_for_ks
from utils import AverageMeterSet, get_exist_path


class Tester(AbstractBaseTrainer):
    def __init__(self,
                 args,
                 model: BaseModel,
                 test_loader: data_utils.DataLoader,
                 device: str,
                 tag: str,
                 accum_iter: int = 0):
        super().__init__(args, device)

        self.model = model
        self.test_loader = test_loader
        self.accum_iter = accum_iter
        self.tag = tag

        assert_model_device(self.model, self.device, self.tag, args.device_idx)

        self.epoch = 0

        # self.loss_fct = BasicLoss()
        # self.iter_per_epoch = len(self.train_loader) * self.batch_size
        # self.tot_iter = self.num_epochs * self.iter_per_epoch

        # logging.info('{} iter per epoch'.format(self.iter_per_epoch))

        self.enable_neg_sample = args.test_negative_sample_size != 0

    @classmethod
    def code(cls):
        return 'tester'

    def calculate_metrics(self, batch) -> dict:
        batch = [x.to(self.device) for x in batch]

        if self.enable_neg_sample:
            logging.fatal("codes for evaluating with negative candidates has bug")
            raise NotImplementedError(
                "codes for evaluating with negative candidates has bug")
            scores = self.model.predict(batch)
        else:
            # seqs, answer, ratings, ... = batch
            seqs = batch[0]
            answer = batch[1]
            ratings = batch[2]

            batch_size = len(seqs)
            labels = torch.zeros(
                batch_size, self.num_items + 1, device=self.device)
            scores = self.model.full_sort_predict(batch)

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

    def test(self, export_root: Path):
        logging.info("Finished loading. Start testing.")

        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.test_loader

            for batch_idx, batch in enumerate(iterator):

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

        average_metrics = average_meter_set.averages()

        if export_root is not None:
            result_folder = get_exist_path(export_root.joinpath(self.tag + "_logs"))
            result_folder.mkdir(exist_ok=True)

            result_file = result_folder.joinpath('test_metrics.json')

            with result_file.open('w') as f:
                json.dump(average_metrics, f)

        return average_metrics        
    
    def train(self):
        pass

    def _train_one_epoch(self):
        pass

    def final_validate(self, export_root: str):
        pass

    def close_training(self):
        pass

    def validate(self):
        pass

    def calculate_loss(self, batch):
        pass

    def test_with_given_state_path(self, state_path):
        pass

    def _create_log_data(self, metrics: dict = None):
        pass

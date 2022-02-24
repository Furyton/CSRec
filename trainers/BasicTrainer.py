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


class Trainer(AbstractBaseTrainer):
    def __init__(self,
                 args,
                 model: BaseModel,
                 optim: optm.Optimizer,
                 lr_sched: optm.lr_scheduler,
                 train_loader: data_utils.DataLoader,
                 val_loader: data_utils.DataLoader,
                 test_loader: data_utils.DataLoader,
                 logger: LoggerService,
                 device: str,
                 tag: str,
                 accum_iter: int = 0):
        super().__init__(args, device)

        self.model = model
        self.optimizer = optim
        self.lr_scheduler = lr_sched
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.accum_iter = accum_iter
        self.tag = tag

        assert_model_device(self.model, self.device, self.tag, args.device_idx)

        self.epoch = 0

        self.loss_fct = BasicLoss()
        self.iter_per_epoch = len(self.train_loader) * self.batch_size
        self.tot_iter = self.num_epochs * self.iter_per_epoch

        logging.info('{} iter per epoch'.format(self.iter_per_epoch))

        self.enable_neg_sample = args.test_negative_sample_size != 0

    @classmethod
    def code(cls):
        return 'trainer'

    def close_training(self):
        log_data = self._create_log_data()
        self.logger.complete(log_data)

        logging.info("finished training")

    def train(self):
        self.validate()
        for self.epoch in range(self.num_epochs):
            logging.info("epoch: " + str(self.epoch))

            t = time.time()

            self._train_one_epoch()
            self.validate()

            logging.info("duration: " + str(time.time() - t) + 's')

        self.close_training()

    def calculate_loss(self, batch):
        batch = [x.to(self.device) for x in batch]
        # seqs, labels, rating = batch

        return self.loss_fct.calculate_loss(self.model, batch)

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_one_epoch(self):
        self.model.train()

        average_meter_set = AverageMeterSet()

        iterator = self.train_loader

        tot_loss = 0.
        tot_batch = 0

        for batch_idx, batch in enumerate(iterator):
            batch_size = batch[0].size(0)

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)

            tot_loss += loss.item()

            tot_batch += 1

            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())

            self.accum_iter += batch_size

            if self._needs_to_log(self.accum_iter):

                log_data = self._create_log_data(average_meter_set.averages())

                self.logger.log_train(log_data)

        logging.info('loss = ' + str(tot_loss / tot_batch))

        self.lr_scheduler.step()

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

    def validate(self):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.val_loader

            for batch_idx, batch in enumerate(iterator):
                # batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

            average_metrics = average_meter_set.averages()
            log_data = self._create_log_data(average_metrics)

            self.logger.log_val(log_data)

            logging.info(average_metrics)
        
        return average_metrics
    
    def final_validate(self, export_root: str):
        logging.info('validate model on val set!')

        state_path = get_best_state_path(export_root, self.tag, must_exist=True)

        load_state_from_given_path(self.model, state_path, self.device, must_exist=True)

        result = self.validate()

        result_folder = get_exist_path(export_root.joinpath(self.tag + "_logs"))

        result_file = result_folder.joinpath('val_metrics.json')

        with result_file.open('w') as f:
            json.dump(result, f)

        return result
        

    def _test(self):
        logging.info(f"Start testing {self.tag}.")

        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.test_loader

            for batch_idx, batch in enumerate(iterator):

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

        average_metrics = average_meter_set.averages()
        logging.info(average_metrics)

        return average_metrics   

    def test(self, export_root: Path):
        logging.info('Test model on test set!')

        state_path = get_best_state_path(export_root, self.tag, must_exist=True)

        result =  self.test_with_given_state_path(state_path)

        result_folder = get_exist_path(export_root.joinpath(self.tag + "_logs"))

        result_file = result_folder.joinpath('test_metrics.json')

        with result_file.open('w') as f:
            json.dump(result, f)

        return result

    def test_with_given_state_path(self, state_path):
        # load_state_from_local(self.model, export_root, self.tag, self.device)
        logging.info(f"Loading model params from {state_path}.")

        load_state_from_given_path(self.model, state_path, self.device, must_exist=True)

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
        logging.info(average_metrics)

        return average_metrics        

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0

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

import logging
import time

import torch
import torch.utils.data as data_utils
from configuration.config import *
from loggers import LoggerService
from models.base import BaseModel
from scheduler.utils import (get_best_state_path, get_state_dict_from,
                             load_state_from_local)

from trainers.BaseTrainer import AbstractBaseTrainer
from trainers.utils import recalls_ndcgs_and_mrr_for_ks
from utils import AverageMeterSet, get_path


class VoteEnsembleTrainer(AbstractBaseTrainer):
    def __init__(self,
                 args,
                 train_loader: data_utils.DataLoader,
                 val_loader: data_utils.DataLoader,
                 test_loader: data_utils.DataLoader,
                 model_list: list,
                 trainer_list: list,
                 tag_list: list,
                 tag: str,
                 logger: LoggerService,
                 device: str):
        super().__init__(args, device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model_list = model_list
        self.trainer_list = trainer_list
        self.tag_list = tag_list
        self.tag = tag
        self.logger = logger

        logging.info(f"There are {len(model_list)} base models to ensemble.")

        self.epoch = 0
        self.accum_iter = 0

        self.debug = 2

    @classmethod
    def code(cls):
        return 'vote_ensemble'

    def close_training(self):
        for trainer in self.trainer_list:
            trainer: AbstractBaseTrainer

            trainer.close_training()

        logging.info("finished training")

    def train(self):
        self.validate()
        for self.epoch in range(self.num_epochs):
            logging.info("Ensemble: epoch: " + str(self.epoch))

            t = time.time()

            self._train_one_epoch()
            self.validate()

            self.accum_iter += 1

            logging.info("duration: " + str(time.time() - t) + 's')

            # self.lr_scheduler.step()

        log_data = self._create_log_data()
        self.logger.complete(log_data)

        self.close_training()

    def calculate_loss(self, batch):
        pass
        # batch = [x.to(self.device) for x in batch]
        # # seqs, labels, rating = batch

        # output = self.model(batch)

        # return self.loss.compute(output, batch)

    # def _get_lr(self):
    #     for param_group in self.optimizer.param_groups:
    #         return param_group['lr']

    def _train_one_epoch(self):
        # self.model.train()
        for idx, (tag, trainer) in enumerate(zip(self.tag_list, self.trainer_list)):
            trainer: AbstractBaseTrainer 

            trainer.epoch = self.epoch

            logging.info(f"train {idx}th model {tag}.")
            
            trainer._train_one_epoch()
            trainer.validate()

        # average_meter_set = AverageMeterSet()

        # iterator = self.train_loader

        # tot_loss = 0.
        # tot_batch = 0

        # for batch_idx, batch in enumerate(iterator):
        #     batch_size = batch[0].size(0)

        #     self.optimizer.zero_grad()
        #     loss = self.calculate_loss(batch)

        #     tot_loss += loss.item()

        #     tot_batch += 1

        #     loss.backward()

        #     self.optimizer.step()

        #     average_meter_set.update('loss', loss.item())

        #     self.accum_iter += batch_size

        #     if self._needs_to_log(self.accum_iter):

        #         log_data = self._create_log_data(average_meter_set.averages())

        #         self.logger.log_train(log_data)

        # logging.info('loss = ' + str(tot_loss / tot_batch))

    @torch.no_grad()
    def early_exit(self, batch):
        threshold = 0.05
        batch_size = batch[0].size(0)

        stacked_pred = torch.stack([ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ], dim=-1)

        cummulate_sum = torch.cumsum(stacked_pred, dim=-1)

        n_model = len(self.model_list)

        confidence = cummulate_sum.max(dim=1).values / (torch.arange(1, n_model + 1, device=self.device)).repeat(batch_size, 1)


        predict = cummulate_sum[torch.arange(batch_size), :, confidence.max(dim=-1).indices]

        if self.debug != 0:
            self.debug -= 1
            logging.debug(f"stacked_pred: {stacked_pred}, size: {stacked_pred.size()}")

            logging.debug(f"cummulate_sum: {cummulate_sum}, size: {cummulate_sum.size()}")

            logging.debug(f"confidence: {confidence}, size: {confidence.size()}")

            logging.debug(f"predict: {predict}, size: {predict.size()}")

        return predict

        # accumulate_predict = [self.model_list[0].full_sort_predict(batch).softmax(dim=-1)]

        # accumulate_predict = self.model_list[0].full_sort_predict(batch).softmax(dim=-1)
        # accumulate_predict: torch.Tensor
        # for model in self.model_list[1:]:
        #     confidence = accumulate_predict.max(dim=-1).values.unsqueeze(-1)
        #     if max(confidence) >= 1 - threshold:
        #         break
        #     predict = model.full_sort_predict(batch).softmax(dim=-1)
        #     accumulate_predict = confidence * accumulate_predict + (1. - confidence) * predict
        
        # return accumulate_predict
    @torch.no_grad()
    def weighted_mix(self, batch):
        predict_list = [ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ]

        predict = torch.stack(predict_list, dim=1)

        confidence = predict.max(dim=-1).values

        weight = confidence / confidence.sum(-1).unsqueeze(-1)

        weighted_predict = (predict * weight.unsqueeze(-1)).sum(1)

        if self.debug != 0:
            self.debug -= 1
            logging.debug(f"predict: {predict}, size: {predict.size()}")

            logging.debug(f"confidence: {confidence}, size: {confidence.size()}")

            logging.debug(f"weight: {weight}, size: {weight.size()}")

            logging.debug(f"weighted_predict: {weighted_predict}, size: {weighted_predict.size()}")

        return weighted_predict

    def get_average_full_sort_predict(self, batch):
        # return self.early_exit(batch)

        # return self.weighted_mix(batch)

        predict = [ model.full_sort_predict(batch).softmax(dim=-1) for model in self.model_list ]
        weight_list = [0.8, 0.2]

        predict = [model.full_sort_predict(batch).softmax(dim=-1) for idx, model in enumerate(self.model_list)]

        w_predict = [_predict * weight_list[idx] for idx, _predict in enumerate(predict)]

        final_predict = sum(w_predict) / len(predict)
        # B x N
        if self.debug != 0:
            self.debug -= 1
            logging.debug("raw predict")
            logging.debug(predict)
            logging.debug("raw confidence")
            logging.debug(f"{[pred.max(dim=-1).values for pred in predict]}")

            logging.debug("weighted predict")
            logging.debug(w_predict)

            logging.debug("final predict")
            logging.debug(final_predict)
            logging.debug("final confidence")
            logging.debug(final_predict.max(dim=-1).values)

        return final_predict

        # return self.early_exit(batch)

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
            scores = self.get_average_full_sort_predict(batch)

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

        metrics = recalls_ndcgs_and_mrr_for_ks(scores, labels, self.metric_ks, ratings)
        return metrics

    def validate(self):
        # self.model.eval()
        logging.info("Validate ensemble model.")
        for model in self.model_list:
            model.eval()

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

    def _test(self):
        for model in self.model_list:
            model: BaseModel

            model.eval()

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

    def test(self, export_root: str):
        logging.info('Test ensemble model on test set!')

        state_path = get_best_state_path(export_root, self.tag, must_exist=True)

        return self.test_with_given_state_path(state_path)

        # model.eval()

        # for model, tag in zip(self.model_list, self.tag_list):
        #     load_state_from_local(model, export_root, tag, self.device, must_exist=True)

        #     model.eval()

        # average_meter_set = AverageMeterSet()

        # with torch.no_grad():
        #     iterator = self.test_loader

        #     for batch_idx, batch in enumerate(iterator):

        #         metrics = self.calculate_metrics(batch)

        #         for k, v in metrics.items():
        #             average_meter_set.update(k, v)

        # average_metrics = average_meter_set.averages()
        # logging.info(average_metrics)

        # return average_metrics

    def test_with_given_state_path(self, state_path):
        _state_path = get_path(state_path)
        state_dict = get_state_dict_from(_state_path, self.device)

        logging.info(f"ensemble model: checkpoint epoch: {state_dict[EPOCH_DICT_KEY]}")

        logging.info(f"ensemble model: loading base models' params.")

        for tag, model in zip(self.tag_list, self.model_list):
            tag: str
            model: BaseModel

            model.load_state_dict(state_dict[tag][STATE_DICT_KEY])

            model.eval()

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
            EPOCH_DICT_KEY: self.epoch,
            ACCUM_ITER_DICT_KEY: self.accum_iter
        }

        for tag, trainer in zip(self.tag_list, self.trainer_list):
            tag: str
            trainer: AbstractBaseTrainer

            data_dict[tag] = trainer._create_log_data()

        if metrics is not None:
            data_dict.update(metrics)

        return data_dict

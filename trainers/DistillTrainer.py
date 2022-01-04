import logging

import torch.optim as optm
import torch.utils.data as data_utils
from configuration.config import *
from loggers import LoggerService

from trainers.loss import SoftLoss
from trainers.Trainer import Trainer
from trainers.utils import assert_model_device


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

        self.model = self.model_list[0]
        self.auxiliary_model = self.model_list[1]

        for model, tag in zip(self.model_list, self.tag_list):
            assert_model_device(model, self.device, tag, args.device_idx)

        self.loss_fct = SoftLoss(self.auxiliary_model, args)

        self.iter_per_epoch = len(self.train_loader) * self.batch_size
        self.tot_iter = self.num_epochs * self.iter_per_epoch

        logging.info('{} iter per epoch'.format(self.iter_per_epoch))


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

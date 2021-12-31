"""
Current target:
__init__:

    generate one dataloader
    generate models
    generate optims
    generate one lr_sched

    generate trainers

    load state dict

    generate an ensembler
_fit:
    create logger service

    dataloader, lr_sched, [(model, optim, trainers, logger, tag), ...]

    take above as input of an ensembler

    tag is the prefix

    train ensembler

_eval:

    test ensembler

set up all these manully without any config

use voting ensembler for now
"""

# from sklearn import ensemble
import logging
import torch
import dataloaders
import models
import trainers
from loggers import (BestModelLogger, LoggerService, MetricGraphPrinter,
                     RecentModelLogger)
from models import MODELS
from torch.utils.tensorboard import SummaryWriter
from trainers import VoteEnsembleTrainer
from trainers.Trainer import Trainer

from scheduler.BaseSched import _BaseSched
from scheduler.utils import (generate_lr_scheduler, generate_model,
                             generate_optim, load_state_from_given_path)
from utils import get_exist_path, get_path
from configuration.config import *


class EnsembleDistillSched(_BaseSched):
    def __init__(self, args, export_root: str) -> None:
        super().__init__()
        self.args = args
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.device = args.device
        self.export_root = get_path(export_root)
        self.mode = args.mode

        self.train_loader, self.val_loader, self.test_loader, self.dataset = dataloaders.dataloader_factory(args)

        self.tag_list = self.model_code_list = ['gru4rec', 'deepfm', 'caser'] #list(MODELS.keys())
        self.tag = VoteEnsembleTrainer.code()

        self.model_list = generate_model(args, self.tag_list, self.dataset, self.device)
        self.optim_list = generate_optim(args, args.optimizer, self.model_list)

        # self.gru4rec_path = "/data/wushiguang-slurm/code/soft-rec/_train/ml_10m_ensemble_cont1_ensemble_caser_gru_2021-12-09_0/gru4rec_logs/checkpoint/best_acc_model.pth"
        # self.caser_path = "/data/wushiguang-slurm/code/soft-rec/_train/ml_10m_ensemble_cont1_ensemble_caser_gru_2021-12-09_0/caser_logs/checkpoint/best_acc_model.pth"
        # self.deepfm_path = "/data/wushiguang-slurm/code/soft-rec/_train/ml_10m_cont1_deepfm_2021-12-11_0/deepfm_logs/checkpoint/best_acc_model.pth"

        # load_state_from_given_path(self.model_list[0], self.gru4rec_path, self.device)
        # load_state_from_given_path(self.model_list[1], self.deepfm_path, self.device)

        # if args.model_state_path is not None:
        #     state_dict = torch.load(args.model_state_path, map_location=torch.device(self.device))

        #     for tag, model, optim in zip(self.tag_list, self.model_list, self.optim_list):
        #         state = state_dict[tag]
        #         model.load_state_dict(state[STATE_DICT_KEY])
        #         optim.load_state_dict(state[OPTIMIZER_STATE_DICT_KEY])		

        #     logging.info(f"checkpoint epoch: {state_dict[EPOCH_DICT_KEY]}")

        self.writer_list = []
        self.logger_list = []

        for tag in self.tag_list:
            _writer, _logger = self._create_logger_service(tag)
            self.writer_list.append(_writer)
            self.logger_list.append(_logger)
        
        self.trainer_list = [
            trainers.trainer_factory(args,
                                     Trainer.code(),
                                     model,
                                     tag,
                                     self.train_loader,
                                     self.val_loader,
                                     self.test_loader,
                                     self.device,
                                     logger,
                                     generate_lr_scheduler(optim, args),
                                     optim) 
            for tag, model, optim, logger in zip(self.tag_list, self.model_list, self.optim_list, self.logger_list)]

        self.ensembler_writer, self.ensembler_logger = self._create_logger_service(self.tag)

        self.ensembler = trainers.trainer_factory(args,
                                                  VoteEnsembleTrainer.code(),
                                                  self.model_list,
                                                  self.tag_list,
                                                  self.train_loader,
                                                  self.val_loader,
                                                  self.test_loader,
                                                  self.device,
                                                  self.ensembler_logger,
                                                  trainer_list=self.trainer_list)

        self.ensembler: VoteEnsembleTrainer
        # self.ensembler = ensemble

    def run(self):
        if self.mode == 'train':
            self._fit()
        self._evaluate()
        self._close_writer()

    def _close_writer(self):
        for writer in self.writer_list + [self.ensembler_writer]:
            writer.close()

    def _fit(self):
        logging.info("Start training ensembler.")

        self.ensembler.train()

    def _evaluate(self):
        if self.mode == "test":
            # for trainer in self.trainer_list:
            #     trainer: Trainer

            #     trainer._test()

            logging.info("evaluate ensembler")
            results = self.ensembler._test()
        else:
            results = self.ensembler.test(self.export_root)

        logging.info(f"!!Final Result!!: {results}")

        # result_folder = self.export_root.joinpath()

    def _create_logger_service(self, prefix: str, metric_only: bool = False):
        """
        Warning:
            Writer should be closed manually.
        """
        _, writer, train_logger, val_logger = self._create_loggers(prefix, metric_only)

        return writer, LoggerService(train_logger, val_logger)

    def _create_loggers(self, prefix: str, metric_only: bool = False):
        """
        desired folder structure

        - experiment
            - train_xx-xx-xx
                - xxx_logs
                    - tb_vis
                    - checkpoint

                    test_results.json
        """
        log_folder = get_exist_path(self.export_root.joinpath(prefix + "_logs"))

        tb_vis_folder = get_exist_path(log_folder.joinpath("tb_vis"))

        writer = SummaryWriter(tb_vis_folder)

        model_checkpoint = get_exist_path(log_folder.joinpath("checkpoint"))

        train_loggers = []

        if not metric_only:
            train_loggers = [
                MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),

                MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train')
            ]

        val_loggers = []

        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation')
            )

            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation')
            )

        if not metric_only:
            val_loggers.append(RecentModelLogger(model_checkpoint))
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))

        return log_folder, writer, train_loggers, val_loggers



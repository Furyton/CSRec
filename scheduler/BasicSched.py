import logging

from dataloaders import dataloader_factory
from loggers import BestModelLogger, LoggerService, MetricGraphPrinter, RecentModelLogger
from torch.utils.tensorboard import SummaryWriter
from trainers import trainer_factory
from trainers.BasicTrainer import Trainer

from scheduler.BaseSched import BaseSched
from scheduler.utils import (generate_lr_scheduler, generate_model,
                             generate_optim, load_state_from_given_path)
from utils import get_exist_path, get_path


class BasicScheduler(BaseSched):
    def __init__(self, args, export_root: str):
        super().__init__()

        self.args = args
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.device = args.device
        self.tag = args.model_code
        self.mode = args.mode  # test or train

        logging.debug(f"BasicScheduler attribs: tag={self.tag}")

        self.export_root = get_path(export_root)

        self.train_loader, self.val_loader, self.test_loader, self.dataset = dataloader_factory(args)

        self.model = generate_model(args, self.tag, self.dataset, self.device)
        self.optim = generate_optim(args, args.optimizer, self.model)

        self.writer, self.logger = self._create_logger_service(self.tag)

        logging.info(str(self.model))

        if args.mode == 'test':
            self.accum_iter = 0
            self.test_state_path = args.test_state_path
        else:
            self.accum_iter = load_state_from_given_path(self.model, args.model_state_path, self.device, self.optim,
                                                         must_exist=False)

        self.trainer = trainer_factory(args,
                                       Trainer.code(),
                                       self.model,
                                       self.tag,
                                       self.train_loader,
                                       self.val_loader,
                                       self.test_loader,
                                       self.device,
                                       self.logger,
                                       generate_lr_scheduler(self.optim, args),
                                       self.optim,
                                       self.accum_iter)

        self.trainer: Trainer

    def run(self):
        if self.mode == 'train':
            self._fit()

        self._evaluate()
        self._close_writer()

    def _close_writer(self):
        self.writer.close()

    def _fit(self):
        logging.info("Start training.")

        self.trainer.train()

    def _evaluate(self):
        if self.mode == 'test':
            results = self.trainer.test_with_given_state_path(self.test_state_path)
        else:
            results = self.trainer.test(self.export_root)

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

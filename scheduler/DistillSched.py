import logging

from dataloaders import dataloader_factory
from loggers import BestModelLogger, LoggerService, MetricGraphPrinter, RecentModelLogger
from torch.utils.tensorboard import SummaryWriter
from scheduler.Routine import Routine
from trainers import trainer_factory
from trainers.DistillTrainer import DistillTrainer
from trainers.BasicTrainer import Trainer

from scheduler.BaseSched import BaseSched
from scheduler.utils import (generate_lr_scheduler, generate_model,
                             generate_optim, load_state_from_given_path)
from utils import get_exist_path, get_path

class DistillScheduler(BaseSched):
    def __init__(self, args, export_root: str):
        super().__init__()

        self.args = args
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.device = args.device
        self.teacher_code = args.mentor_code
        self.model_code = args.model_code
        self.mode = args.mode # test or train
        self.test_state_path = args.test_state_path

        self.teacher_tag = "teacher_" + self.teacher_code
        self.model_tag = "student_" + self.model_code

        logging.debug(f"DistillScheduler attribs: teacher tag={self.teacher_tag}, student tag={self.model_tag}")

        self.export_root = get_path(export_root)

        self.train_loader, self.val_loader, self.test_loader, self.dataset = dataloader_factory(args)

        # teacher

        self.teacher = generate_model(args, self.teacher_code, self.dataset, self.device)
        self.t_optimizer = generate_optim(args, args.optimizer, self.teacher)

        self.t_writer, self.t_logger = self._create_logger_service(self.teacher_tag)

        self.t_accum_iter = load_state_from_given_path(self.teacher, args.mentor_state_path, self.device, self.t_optimizer, must_exist=False)

        logging.debug("teacher model: \n" + str(self.teacher))

        self.t_trainer = trainer_factory(args,
                                Trainer.code(),
                                self.teacher,
                                self.teacher_tag,
                                self.train_loader,
                                self.val_loader,
                                self.test_loader,
                                self.device,
                                self.t_logger,
                                generate_lr_scheduler(self.t_optimizer, args),
                                self.t_optimizer,
                                self.t_accum_iter)

        # student

        self.student = generate_model(args, self.model_code, self.dataset, self.device)
        self.s_optimizer = generate_optim(args, args.optimizer, self.student)

        self.s_writer, self.s_logger = self._create_logger_service(self.model_tag)

        self.s_accum_iter = load_state_from_given_path(self.student, args.model_state_path, self.device, self.s_optimizer, must_exist=False)

        logging.debug("student model: \n" + str(self.student))

        self.s_trainer = trainer_factory(args,
                                DistillTrainer.code(),
                                [self.student, self.teacher],
                                [self.model_tag, self.teacher_tag],
                                self.train_loader,
                                self.val_loader,
                                self.test_loader,
                                self.device,
                                self.s_logger,
                                generate_lr_scheduler(self.s_optimizer, args),
                                self.s_optimizer,
                                self.s_accum_iter)

        self.t_trainer: Trainer
        self.s_trainer: DistillTrainer

        self.routine = Routine(['teacher', 'student'], [self.t_trainer, self.s_trainer], self.args, self.export_root)

    def run(self):
        return super().run()

    def _finishing(self):
        self.t_writer.close()
        self.s_writer.close()

    def _fit(self):
        self.routine.run_routine()

    def _evaluate(self):
        logging.info("Start testing student model on test set")

        if self.test_state_path is not None:
            results = self.s_trainer.test_with_given_state_path(self.test_state_path)
        else:
            results = self.s_trainer.test(self.export_root)

        logging.info(f"!!Final Result!!: {results}")

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



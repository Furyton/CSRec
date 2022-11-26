from copy import deepcopy
import logging
import torch
import dataloaders
import models
from models.Ensembler import Ensembler
from scheduler.Routine import Routine
import trainers
from loggers import (BestModelLogger, LoggerService, MetricGraphPrinter,
                     RecentModelLogger)
from models import MODELS
from torch.utils.tensorboard import SummaryWriter
from trainers import trainer_factory
from trainers.BasicTrainer import Trainer

from scheduler.BaseSched import BaseSched
from scheduler.utils import (generate_lr_scheduler, generate_model,
                             generate_optim, load_state_from_given_path, model_path_finder)
from trainers.DistillTrainer import DistillTrainer
from utils import get_exist_path, get_path
from configuration.config import *


class EnsemblePartialDistillSched(BaseSched):
    def __init__(self, args, export_root: str) -> None:
        super().__init__()
        self.args = args
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.device = args.device
        self.export_root = get_path(export_root)
        self.mode = args.mode
        self.test_state_path = args.test_state_path

        self.teacher1_code = args.mentor_code
        self.teacher2_code = args.mentor2_code
        self.student_code = args.model_code

        self.weight_list = args.weight_list
        self.temperature = args.T

        self.teacher1_tag = "teacher1_" + self.teacher1_code
        self.teacher2_tag = "teacher2_" + self.teacher2_code
        self.student_tag = "student_" + self.student_code

        self.train_loader, self.val_loader, self.test_loader, self.dataset = dataloaders.dataloader_factory(args)

        if args.enable_auto_path_finder:
            mentor_seed1 = args.mentor_seed1
            mentor_seed2 = args.mentor_seed2

            mentor_base_path = args.mentor_state_path

            mentor_path_pattern = args.mentor_describe

            arg_dict = dict(args._get_kwargs())
            arg_dict["sample_seed"] = mentor_seed1
            arg_dict["model_code"] = self.teacher1_code
            self.args.mentor_state_path = model_path_finder(mentor_base_path, mentor_path_pattern, arg_dict, self.teacher1_code)

            arg_dict["sample_seed"] = mentor_seed2
            arg_dict["model_code"] = self.teacher2_code
            self.args.mentor2_state_path = model_path_finder(mentor_base_path, mentor_path_pattern, arg_dict, self.teacher2_code)


        self.teacher1, self.t_trainer1, self.t_writer1 = self._generate_teacher_trainer(self.teacher1_code, self.teacher1_tag, self.args.mentor_state_path)

        self.teacher2, self.t_trainer2, self.t_writer2 = self._generate_teacher_trainer(self.teacher2_code, self.teacher2_tag, self.args.mentor2_state_path)

        self._generate_student_trainer()

        self.routine = Routine(["teacher1", "teacher2", "student"], [self.t_trainer1, self.t_trainer2, self.s_trainer], self.args, self.export_root)

    def _gen_teacher_args(self, seed: int):
        new_args = deepcopy(self.args)
        new_args.do_sampling = True
        new_args.load_processed_dataset = new_args.load_sampled_dataset
        new_args.save_processed_dataset = new_args.save_sampled_dataset
        new_args.sample_seed = seed

        return new_args
    
    def _gen_teacher_dataloader(self, seed: int):
        new_args = self._gen_teacher_args(seed)

        return dataloaders.dataloader_factory(new_args)

    def _generate_teacher_trainer(self, code: str, tag: str, seed: int, state_path: str=None):

        tr, val, ts, dataset = self._gen_teacher_dataloader(seed)

        teacher = generate_model(self.args, code, dataset, self.device)

        optimizer = generate_optim(self.args, self.args.optimizer, teacher)

        writer, logger = self._create_logger_service(tag)

        accum_iter = load_state_from_given_path(teacher, state_path, self.device, optimizer, must_exist=False)

        logging.debug(f"{tag}: \n{teacher}")

        trainer = trainer_factory(self.args,
                        Trainer.code(),
                        teacher,
                        tag,
                        tr,
                        val,
                        ts,
                        self.device,
                        logger,
                        generate_lr_scheduler(optimizer, self.args),
                        optimizer,
                        accum_iter)

        return teacher, trainer, writer

    def _generate_student_trainer(self):
        self.student = generate_model(self.args, self.student_code, self.dataset, self.device)

        self.s_optimizer = generate_optim(self.args, self.args.optimizer, self.student)

        self.s_writer, self.s_logger = self._create_logger_service(self.student_tag)

        self.s_accum_iter = load_state_from_given_path(self.student, self.args.model_state_path, self.device, self.s_optimizer, must_exist=False)

        self.mix_teacher = Ensembler(self.device, [self.teacher1, self.teacher2], self.weight_list, self.temperature)

        self.mix_teacher_tag = "mix_" + self.teacher1_code + "_" + self.teacher2_code

        logging.debug(f"{self.student_tag}: \n{self.student}")

        
        self.s_trainer = trainer_factory(self.args,
                                DistillTrainer.code(),
                                [self.student, self.mix_teacher],
                                [self.student_tag, self.mix_teacher_tag],
                                self.train_loader,
                                self.val_loader,
                                self.test_loader,
                                self.device,
                                self.s_logger,
                                generate_lr_scheduler(self.s_optimizer, self.args),
                                self.s_optimizer,
                                self.s_accum_iter)

        
    def run(self):
        return super().run()

    def _finishing(self):
        self.t_writer1.close()
        self.t_writer2.close()
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



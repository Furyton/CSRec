from copy import deepcopy
import logging

from dataloaders import dataloader_factory
from loggers import (BestModelLogger, LoggerService, MetricGraphPrinter,
                     RecentModelLogger)
from torch.utils.tensorboard import SummaryWriter
from trainers import BaseTrainer, trainer_factory
from trainers.BasicTrainer import Trainer
from trainers.DVAETrainer import DVAETrainer
from trainers.DistillTrainer import DistillTrainer

from scheduler.BaseSched import BaseSched
from scheduler.Routine import Routine
from scheduler.utils import (generate_lr_scheduler, generate_model,
                             generate_optim, load_state_from_given_path, model_path_finder)
from utils import get_exist_path, get_path


class DVAEDistillScheduler(BaseSched):
    def __init__(self, args, export_root: str):
        super().__init__()

        self.args = args
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.device = args.device
        self.auxiliary_code = args.mentor_code
        self.teacher_code = args.mentor2_code
        self.model_code = args.model_code
        self.mode = args.mode # test or train
        self.test_state_path = args.test_state_path

        self.auxiliary_tag = "auxiliary_" + self.auxiliary_code
        self.teacher_tag = "teacher_" + self.teacher_code
        self.model_tag = "student_" + self.model_code

        logging.debug(f"DVAEDistillScheduler attribs: auxiliary tag={self.auxiliary_tag}, teacher tag={self.teacher_tag}, student tag={self.model_tag}")

        self.export_root = get_path(export_root)

        self.train_loader, self.val_loader, self.test_loader, self.dataset = dataloader_factory(args)

        if args.enable_auto_path_finder:
            # find model state path for auxiliary model
            mentor_seed = args.sample_seed
            mentor_base_path = args.mentor_state_path

            mentor_path_pattern = args.mentor_describe

            arg_dict = dict(args._get_kwargs())
            arg_dict["sample_seed"] = mentor_seed
            arg_dict["model_code"] = self.auxiliary_code
            self.args.mentor_state_path = model_path_finder(mentor_base_path, mentor_path_pattern, arg_dict, self.auxiliary_code)

        self._generate_auxliary_trainer(args.sample_seed)
        self._generate_teacher_trainer()
        self._genearte_student_trainer()

        self.routine = Routine(['auxiliary', 'teacher', 'student'], [self.a_trainer, self.t_trainer, self.s_trainer], self.args, self.export_root)

    def run(self):
        return super().run()

    def _finishing(self):
        self.a_writer.close()
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

    def _generate_auxiliary_args(self, seed: int):
        new_args = deepcopy(self.args)
        new_args.do_sampling = True
        new_args.load_processed_dataset = new_args.load_sampled_dataset
        new_args.save_processed_dataset = new_args.save_sampled_dataset
        new_args.sample_seed = seed

        return new_args
    
    def _gen_auxiliary_dataloader(self, seed: int):
        new_args = self._generate_auxiliary_args(seed)

        return dataloader_factory(new_args)

    def _generate_auxliary_trainer(self, seed: int):
        logging.info(f"generating dataset (sampled with seed {seed}) and dataloader")

        tr, val, ts, dataset = self._gen_auxiliary_dataloader(seed)

        self.auxiliary = generate_model(self.args, self.auxiliary_code, dataset, self.device)

        self.a_optimizer = generate_optim(self.args, self.args.optimizer, self.auxiliary)

        self.a_writer, self.a_logger = self._create_logger_service(self.auxiliary_tag)

        self.a_accum_iter = load_state_from_given_path(self.auxiliary, self.args.mentor_state_path, self.device, self.a_optimizer, must_exist=False)

        logging.debug("auxiliary model: \n" + str(self.auxiliary))

        self.a_trainer = trainer_factory(self.args,
                        Trainer.code(),
                        self.auxiliary,
                        self.auxiliary_tag,
                        tr,
                        val,
                        ts,
                        self.device,
                        self.a_logger,
                        generate_lr_scheduler(self.a_optimizer, self.args),
                        self.a_optimizer,
                        self.a_accum_iter)

        self.a_trainer: BaseTrainer

    def _generate_teacher_trainer(self):
        self.prior = generate_model(self.args, 'prior', self.dataset, self.device)

        self.teacher = generate_model(self.args, self.teacher_code, self.dataset, self.device)

        self.t_optimizer = generate_optim(self.args, self.args.optimizer, [self.prior, self.teacher], one_optim=True)

        self.t_writer, self.t_logger = self._create_logger_service(self.teacher_tag)

        self.t_accum_iter = load_state_from_given_path(self.teacher, self.args.mentor2_state_path, self.device, self.t_optimizer, must_exist=False)

        # self.t_accum_iter = 0

        logging.debug("prior model: \n" + str(self.prior))

        logging.debug("teacher model: \n" + str(self.teacher))

        self.t_trainer = trainer_factory(self.args,
                        DVAETrainer.code(),
                        [self.teacher, self.auxiliary, self.prior],
                        [self.teacher_tag, self.auxiliary_tag, 'prior'],
                        self.train_loader,
                        self.val_loader,
                        self.test_loader,
                        self.device,
                        self.t_logger,
                        generate_lr_scheduler(self.t_optimizer, self.args),
                        self.t_optimizer,
                        self.t_accum_iter)
        
        self.t_trainer: DVAETrainer

    def _genearte_student_trainer(self):

        self.student = generate_model(self.args, self.model_code, self.dataset, self.device)
        self.s_optimizer = generate_optim(self.args, self.args.optimizer, self.student)

        self.s_writer, self.s_logger = self._create_logger_service(self.model_tag)

        self.s_accum_iter = load_state_from_given_path(self.student, self.args.model_state_path, self.device, self.s_optimizer, must_exist=False)

        logging.debug("student model: \n" + str(self.student))

        self.s_trainer = trainer_factory(self.args,
                                DistillTrainer.code(),
                                [self.student, self.teacher],
                                [self.model_tag, self.teacher_tag],
                                self.train_loader,
                                self.val_loader,
                                self.test_loader,
                                self.device,
                                self.s_logger,
                                generate_lr_scheduler(self.s_optimizer, self.args),
                                self.s_optimizer,
                                self.s_accum_iter)

        self.s_trainer: BaseTrainer

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



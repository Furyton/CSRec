# from configuration.config import FINE_TUNE_STAGE, NORMAL_STAGE, PRETRAIN_STAGE
# from trainers.NormalTrainer import NormalTrainer



import logging

import torch.optim as optm
import torch.utils.data as data_utils
from loggers import LoggerService
from models.base import BaseModel

from trainers.DistillTrainer import DistillTrainer
from trainers.DVAETrainer import DVAETrainer
from trainers.Trainer import Trainer
from trainers.VoteEnsembleTrainer import VoteEnsembleTrainer

TRAINERS = {
    Trainer.code(): Trainer,
    VoteEnsembleTrainer.code(): VoteEnsembleTrainer,
    DistillTrainer.code(): DistillTrainer,
    DVAETrainer.code(): DVAETrainer,
}


# def trainer_factory(args, model, train_loader, val_loader, test_loader, dataset, export_root):
#     return NormalTrainer(args, model, train_loader, val_loader, test_loader, export_root)

def trainer_factory(args,
                    trainer_code: str,
                    model_or_model_list,
                    tag_or_tag_list,
                    train_loader: data_utils.DataLoader,
                    val_loader: data_utils.DataLoader,
                    test_loader: data_utils.DataLoader,
                    device: str,
                    logger: LoggerService,
                    lr_sched: optm.lr_scheduler = None,
                    optim: optm.Optimizer = None,
                    accum_iter: int = 0,
                    trainer_list: list = None):
    trainer = TRAINERS[trainer_code]

    if trainer_code.lower() == Trainer.code():
        assert(isinstance(model_or_model_list, BaseModel) and isinstance(tag_or_tag_list, str) and lr_sched and optim and logger)

        return trainer( args, 
                        model_or_model_list, 
                        optim, 
                        lr_sched, 
                        train_loader, 
                        val_loader, 
                        test_loader, 
                        logger, 
                        device, 
                        tag_or_tag_list, 
                        accum_iter)

    if trainer_code.lower() == VoteEnsembleTrainer.code():
        assert(isinstance(model_or_model_list, list) and isinstance(tag_or_tag_list, list) and trainer_list)

        return trainer( args,
                        train_loader,
                        val_loader,
                        test_loader,
                        model_or_model_list,
                        trainer_list,
                        tag_or_tag_list,
                        VoteEnsembleTrainer.code(),
                        logger,
                        device)
    
    if trainer_code.lower() in [DVAETrainer.code(), DistillTrainer.code()]:
        assert(isinstance(model_or_model_list, list) and isinstance(tag_or_tag_list, list) and lr_sched and optim and logger)

        return trainer( args,
                        optim,
                        lr_sched,
                        train_loader,
                        val_loader,
                        test_loader,
                        model_or_model_list,
                        tag_or_tag_list,
                        logger,
                        device,
                        accum_iter)

    logging.fatal(f"{trainer_code} has not been implemented yet.")

    raise NotImplementedError

# def trainer_factory(args, model, train_loader, val_loader, test_loader, dataset, export_root, mode:str, mentor=None):
#     trainer = TRAINERS[mode]

#     if mode.lower() == PRETRAIN_STAGE:
#         return trainer(args, model, train_loader, val_loader, test_loader, export_root, 'mentor_models')
#     elif mode.lower() == FINE_TUNE_STAGE:
#         return trainer(args, model, train_loader, val_loader, test_loader, export_root, mentor)
#     elif mode.lower() == NORMAL_STAGE:
#         return trainer(args, model, train_loader, val_loader, test_loader, export_root, 'models')
#     else:
#         raise ValueError

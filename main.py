import argparse
import json
from os import path

from configuration.options import args, parser
from scheduler.BasicSched import BasicScheduler
from scheduler.DistillSched import DistillScheduler
from scheduler.EnsembleSched import EnsembleScheduler
from utils import *

# def next_stage(cur_mode: str):
#     NEXT_STAGE = {NORMAL_STAGE: FINISH_STAGE,
#                   PRETRAIN_STAGE: FINE_TUNE_STAGE,
#                   FINE_TUNE_STAGE: FINISH_STAGE}
#     return NEXT_STAGE[cur_mode.lower()]

# def model_factory(args, mode: str, dataset):
#     if args.enable_kd:
#         if mode.lower() == NORMAL_STAGE:
#             raise ValueError

#         if mode.lower() == PRETRAIN_STAGE:
#             return models.model_factory(args, args.mentor_model, dataset), None
#         else:
#             return models.model_factory(args, args.model_code, dataset), models.model_factory(args, args.mentor_model, dataset)
        
#     else:
#         if mode.lower() == PRETRAIN_STAGE:
#             raise ValueError
        
#         if mode.lower() == NORMAL_STAGE:
#             return models.model_factory(args, args.model_code, dataset), None
#         else:
#             return models.model_factory(args, args.model_code, dataset), models.model_factory(args, args.mentor_model, dataset)

# def training_stage_loop(args, mode: str, export_root: str, only_test=False):
#     print('====================')
#     print('training stage mode: {}\nstart...'.format(mode))

#     train_loader, val_loader, test_loader, dataset = dataloader_factory(args, export_root)

#     model, teacher = model_factory(args, mode, dataset)

#     trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, dataset, export_root, mode, teacher)

#     if not only_test:
#         trainer.train()

#     trainer.test()

#     print('finished\ntraining stage mode: {}'.format(mode))
#     print('====================')

# def train():
#     export_root = setup_train(args)

#     cur_mode = args.training_stage

#     while cur_mode != FINISH_STAGE:
#         training_stage_loop(args, cur_mode, export_root)
#         cur_mode = next_stage(cur_mode)

#     return export_root

# def train(args):
#     export_root = setup_train(args)
#     logging.info("start training")

#     train_loader, val_loader, test_loader, dataset = dataloaders.dataloader_factory(args, export_root)

#     model = models.model_factory(args, args.model_code, dataset)

#     trainer = trainers.trainer_factory(args, model, train_loader, val_loader, test_loader, dataset, export_root)

#     trainer.train()
#     logging.info("finished training")

#     trainer.test()
    
#     return export_root

# def test(args):
#     export_root = setup_train(args)
    
#     # cur_mode = args.training_stage

#     # training_stage_loop(args, cur_mode, export_root, True)

#     return export_root


if __name__ == '__main__':
    with open(path.normpath(args.config_file), 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    export_root = setup_train(args)

    # sched = BasicScheduler(args, export_root)
    # sched = EnsembleScheduler(args, export_root)
    sched = DistillScheduler(args, export_root)

    sched.run()
    # if args.mode == 'train':
    #     _ = train(args)
    # elif args.mode == 'test':
    #     _ = test(args)
    # else:
    #     raise ValueError

    # source = args.slurm_log_file_path
    # target= export_r
    # try:
    #     shutil.copy(source, target)
    # except IOError as e:
    #     print("Unable to copy file. %s" % e)
    # except:
    #     print("Unexpected error:", sys.exc_info())

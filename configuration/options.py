from configuration.config import *
from models import MODELS
from dataloaders import DATALOADERS

import argparse

parser = argparse.ArgumentParser(description='furyton')

################
# Top Level
################
parser.add_argument('--config_file', type=str, default='config.json', help="config file for trainer and dataloaders")
parser.add_argument('--mode', nargs='+', type=str, default='train')
parser.add_argument('--rand_seed', type=int, default=2021, help="random seed for all")
parser.add_argument('--task_id', type=int, default=-1)
parser.add_argument('--describe', type=str, default=None)
################
# Test
################
parser.add_argument('--test_state_path', type=str, default=None, help="model state dict path for test")

parser.add_argument('--model_state_path', type=str, default=None, help="model state dict path for training")

# parser.add_argument('--mentor_state_path', type=str, default=None, help="mentor model state dict path for training")
################
# Dataset
################
parser.add_argument('--load_processed_dataset', type=bool, default=False)
parser.add_argument('--save_processed_dataset', type=bool, default=True)
parser.add_argument('--dataset_cache_filename', type=str)

parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--min_length', type=int, default=3, help='minimum length for each user')
parser.add_argument('--min_item_inter', type=int, default=5, help='minimum interaction for each item')
parser.add_argument('--good_only', type=bool, default=True, help='only use items user likes')
parser.add_argument('--do_remap', type=bool, default=True, help="remap the use_id and item_id")
parser.add_argument('--use_rating', type=bool, default=True, help="use rating as a feature in dataloaders")

parser.add_argument('--do_sampling', type=bool, default=False, help="Whether use the subset of the dataset for training. Note: #user and #item will be kept. Actually we just sample users.")
parser.add_argument('--path_for_sample', type=str, default=None)
parser.add_argument('--sample_rate', type=float, default=0.5, help='use sample_rate * #user. Available only if `do_sampling` is True.')
parser.add_argument('--sample_seed', type=int, default=0, help='random sample seed. Available only if `do_sampling` is True. NOTE: This is a LOCAL seed only for sampling.')

################
# Dataloader
################
parser.add_argument('--dataloader_type', type=str, default='mask', choices=DATALOADERS.keys())
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--prop_sliding_window', type=float, default=0.1, help='-1.0 means taking max_len as sliding step, 0~1 indicates prop, >1 means sliding step')
parser.add_argument('--worker_number', type=int, default=1)
################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
# parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100, help="zero means taking the full item set as negative samples")
# parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0') # [0, 1, 2 ... ]
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
# processing #
parser.add_argument('--show_process_bar', type=bool, default=False, help='whether show the processing bar or not')
################
# Model
################
# parser.add_argument('--enable_mentor', type=bool)
# parser.add_argument('--mentor_model', type=str, default='pop', choices=MODELS.keys())

# parser.add_argument('--enable_sample', type=bool, default=False)
# parser.add_argument('--samples_ratio', type=float, default=0.5)

parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--mentor_code', type=str, default='bert', choices=MODELS.keys())
# parser.add_argument('--model_init_seed', type=int, default=None)
parser.add_argument('--max_len', type=int, default=50, help='Length of sequence, better preserve the same for all models for the sake of faireness')

# parser.add_argument('--training_stage', type=str, default=NORMAL_STAGE, choices=[PRETRAIN_STAGE, FINE_TUNE_STAGE, NORMAL_STAGE])

parser.add_argument('--training_routine', type=str, default=None)

# SOFT REC #
# parser.add_argument('--enable_kd', type=bool, default=False, help='Use knowledge distillation')
parser.add_argument('--T', type=float, default=1, help='temperature')
parser.add_argument('--alpha', type=float, default=0.1, help='trade off between original loss and KL div')
parser.add_argument('--dvae_alpha', type=float, default=0.5)
parser.add_argument('--softmaxed_mentor', type=bool, default=False)
################

parser.add_argument('--enable_auto_path_finder', type=bool, default=False)
parser.add_argument('--use_sampled_mentor', type=bool, default=False)
parser.add_argument('--weight_list', nargs='+', type=float, default=None)
parser.add_argument('--mentor_code_list', nargs='+', type=str, default=None)
parser.add_argument('--mentor_path_list', nargs='+', type=str, default=None)
parser.add_argument('--mentor_describe', type=str, default=None)
################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='train')
parser.add_argument('--dataset_name', type=str, default="ml-10m.csv")
# parser.add_argument('--subdataset_rate', type=float, default=0.1)

parser.add_argument('--validation_rate', type=float, default=0.2)

parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--start_index', type=int, default=1)

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = eval(value)
            # TODO

parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, help='usage: -k a=some b=thing, store as a dict')

# if using slurm

# parser.add_argument('--slurm_log_file_path', type=str, default=None)

################
args = parser.parse_args()


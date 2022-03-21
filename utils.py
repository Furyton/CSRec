import contextlib
import json
import logging
import os
import pprint as pp
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim

from configuration.config import *


def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)
    setup_logging(args, export_root)

    # pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)

    logging.info(json.dumps(vars(args), indent=4))

    fix_random_seed_as(args.rand_seed)

    return export_root


def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    os.makedirs(experiment_dir, exist_ok=True)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description, detailed_desc=_get_detailed_description(args), task_id=args.task_id)
    os.makedirs(experiment_path, exist_ok=True)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path

def _get_detailed_description(args):
    if args.describe is None:
        return args.model_code
    else:
        arg_dict = dict(args._get_kwargs())
        container = args.describe
        detailed_desc = args.model_code
        try:
            detailed_desc = container.format(config=arg_dict)
        except:
            detailed_desc = args.model_code
            print(f'Illegal description format{args.describe}, use {args.model_code} instead.')
        return detailed_desc

def get_name_of_experiment_path(experiment_dir, experiment_description, detailed_desc="", task_id=-1):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + detailed_desc + "_" + str(datetime.now().strftime("%m-%d_%H:%M"))))
    if task_id != -1:
        experiment_path = experiment_path + "_t" + str(task_id)
    else:
        idx = _get_experiment_index(experiment_path)
        experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def load_weights(model, path):
    pass


# def save_test_result(export_root, result):
#     filepath = Path(export_root).joinpath('test_result.txt')
#     with filepath.open('w') as f:
#         json.dump(result, f, indent=2)

def setup_logging(args, experiment_path):
    """
    Warning:
        Please don't use logging before finishing setup
    """

    logging_file = os.path.join(experiment_path, args.experiment_description + '.log')

    logging.basicConfig(filename=logging_file, level=logging.DEBUG, format='%(levelname)-8s%(asctime)s %(funcName)s in %(module)s:\n%(message)s', datefmt='[%m-%d %H:%M:%S]', filemode='w')

def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(seed):
    # random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    # np.random.seed(random_seed)
    # cudnn.deterministic = True
    cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

@contextlib.contextmanager
def set_temporary_np_seed_as(seed):
    """
    Temporarily set the numpy rand seed within a context

    Example:
    >>> np.random.seed(0)
    >>> np.random.randn(3)
    array([1.76405235, 0.40015721, 0.97873798])
    >>> np.random.randn(3)
    array([ 2.2408932 ,  1.86755799, -0.97727788])

    >>> np.random.seed(0)
    >>> np.random.randn(3)
    array([1.76405235, 0.40015721, 0.97873798])
    >>> with temp_seed(5):
    ...     np.random.randn(3)
    array([ 0.44122749, -0.33087015,  2.43077119])
    >>> np.random.randn(3)
    array([ 2.2408932 ,  1.86755799, -0.97727788])

    reference:
    https://stackoverflow.com/a/49557127
    """

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def set_up_gpu(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))

def get_exist_path(path) -> Path:

    if path is None:
        logging.fatal("get exist path, but given None.")
        raise ValueError

    p = Path(path)

    if not p.exists():
        if p.is_file():
            logging.warning(f"file {p} does not exist! Create one.")
            p.touch(exist_ok=True)
        else:
            logging.warning(f"dir {p} does not exist! Create one.")
            p.mkdir(exist_ok=True, parents=True)

    return p

def get_path(path) -> Path:
    """
    input:
        path -> str or pathlib.Path
    return:
        pathlib.Path or None

    if path does not exist, a FileNotFoundError will raise.
    if path is None, returns None
    """

    if path is None:
        logging.warning("Path is None")
        return None

    p = Path(path)

    if not p.exists():
        logging.fatal(f"file {p} does not exist!!!")
        raise FileNotFoundError
    
    return p

# for test
# def load_pretrained_weights(model, path, device):
#     chk_dict = torch.load(os.path.abspath(path), map_location=torch.device(device))
#     model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
#     model.load_state_dict(model_state_dict)


# for resume training
# def setup_to_resume(args, model, optimizer):
#     chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training), 'models/checkpoint-recent.pth'))
#     model.load_state_dict(chk_dict[STATE_DICT_KEY])
#     optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])


# def create_optimizer(model, args):
#     if args.optimizer == 'Adam':
#         return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

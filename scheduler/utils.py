"""
model factory
trainer factory
dataloader factory
optimizer factory

state dict loader
"""

import argparse
import collections
import logging
from pathlib import Path
from typing import Iterator

import torch
from torch.nn.parameter import Parameter
from configuration.config import *
from dataloaders import dataloader_factory
from models import model_factory
from models.base import BaseModel

from utils import get_path

from itertools import chain

def generate_dataloader(args):
    return dataloader_factory(args)



def generate_model( args: argparse.Namespace, 
                    model_or_model_list, 
                    dataset: list, 
                    device: str,
                    n_model: int = 1):
    r"""
    Args:
        args: global config

        model_or_model_list: model_code -> str, or a list or tuple of model_code s

        dataset: dataset list

        device: 'cuda' or 'cpu'

        n_model: number of models, default: 1
    return:
        if given a single model_code -> str, and n_model = 1, then return a single model -> BaseModel

        if given a single model_code -> str, n_model > 1, returns a list of model

        if given a list of model_code, returns coresponding list of model
    """

    assert(n_model > 0)

    model_code_list: list
    is_model_list: bool

    if isinstance(model_or_model_list, (str, )):
        is_model_list = (n_model != 1)

        model_code_list = [model_or_model_list] * n_model
    elif isinstance(model_or_model_list, collections.Iterable):
        model_code_list = list(model_or_model_list)
        is_model_list = True
    else:
        logging.fatal(f"illegal args: model_or_model_list. expected str or Iterable, but got {type(model_or_model_list)}")
        raise ValueError

    logging.debug(f"model_code_list={model_code_list}")

    model_list = [] 
    for model_code in model_code_list:
        model_list.append(model_factory(args, model_code, dataset).to(device))
    
    return model_list if is_model_list else model_list[0]



def generate_optim( args: argparse.Namespace,
                    optim_code_or_list, 
                    models,
                    one_optim: bool=False):
    r"""
    Args:
        args: global config

        optim_code_or_list: optimizer_code -> str, or a list of optimizer_code s

        models: a single model->BaseModel or a list of models

        one_optim: default False, if #optim = 1, #models > 1, optimize all `models`'s parameters in one optim
    return:
        the same logic as `generate_model`, except for `n_optim = len(models)`
    """

    if isinstance(models, BaseModel):
        model_list = [models]
    elif isinstance(models, collections.Iterable):
        model_list = list(models)
    else:
        logging.fatal(f"{models} is not acceptable.")
        raise ValueError

    if one_optim:
        n_optim = 1
    else:
        n_optim = len(model_list)

    optim_code_list: list
    is_list: bool

    if isinstance(optim_code_or_list, (str, )):
        is_list = (n_optim != 1) and not one_optim
 
        optim_code_list = [optim_code_or_list] * n_optim
    elif isinstance(optim_code_or_list, collections.Iterable):
        assert(not one_optim or (one_optim and len(optim_code_or_list) == 1))
        optim_code_list = list(optim_code_or_list)

        is_list = True
    else:
        logging.fatal(f"illegal args: optim_code_list. expected str or Iterable obj of str, but got {type(optim_code_list)}")
        raise ValueError

    optim_list = []

    def _gen_optim(code: str, params: Iterator[Parameter]):
        if code.lower() == "adam":
            return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif code.lower() == "sgd":
            return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            logging.fatal(f"{code} optimizer has not been implemented yed.")
            raise NotImplementedError

    if one_optim:
        params = [model.parameters() for model in model_list]
        return _gen_optim(optim_code_list[0], chain(*params))

    for optim_code, model in zip(optim_code_list, model_list):
        optim_list.append(_gen_optim(optim_code, model.parameters()))
    
    return optim_list if is_list else optim_list[0]


def generate_lr_scheduler(optim: torch.optim.Optimizer, args: argparse.Namespace):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.gamma)

    return lr_scheduler

def get_best_state_path(export_root, tag: str, must_exist: bool = False):
    try:
        root = get_path(export_root)
        log_folder = get_path(root.joinpath(tag + '_logs'))
        checkpoint_folder = get_path(log_folder.joinpath('checkpoint'))
        checkpoint_path = get_path(checkpoint_folder.joinpath('best_acc_model.pth'))

        return checkpoint_path
    except:
        if must_exist:
            logging.fatal(f"best state path at {export_root}/{tag}_logs/checkpoint/best_acc_model.pth not found")

            raise FileNotFoundError

def get_state_dict_from(state_path, device):
    return torch.load(state_path, map_location=torch.device(device))

def load_state_from_local(model: BaseModel, export_root: str, tag: str, device: str, optim: torch.optim.Optimizer = None, must_exist: bool = False):
    """
    return: 
        accumulate iter
    """
    
    checkpoint_path = get_best_state_path(export_root, tag, must_exist=must_exist)

    return load_state_from_given_path(model, optim, str(checkpoint_path), device, must_exist=must_exist)

def load_state_from_given_path(model: BaseModel, state_path: str, device: str, optim: torch.optim.Optimizer = None, must_exist: bool = False):
    """
    return: 
        accumulate iter
    """

    try:
        checkpoint_path = get_path(state_path)
    except:
        if must_exist:
            logging.fatal(f"state file at {state_path} should exist.")

            raise FileNotFoundError

        logging.warning(f"{state_path} not exist. I won't load anything.")

        return 0

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        logging.info(f"checkpoint epoch: {checkpoint[EPOCH_DICT_KEY]}")

        logging.info("Loading model's parameters")

        model.load_state_dict(checkpoint[STATE_DICT_KEY])

        if optim is not None:
            logging.info("Loading optimizer's parameters")

            optim.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT_KEY])
        
        return checkpoint[ACCUM_ITER_DICT_KEY]
    else:
        logging.warning("Not given any path.")

        return 0

def model_path_finder(base_path: str, target_pattern: str, filler_dict: dict, tag: str):
    base_root = get_path(base_path)
    target_substring = target_pattern.format(config=filler_dict)
    
    for child in base_root.iterdir():
        if target_substring in child.name:
            return get_best_state_path(child, tag, must_exist=True)
import logging
import os
from abc import ABCMeta, abstractmethod

import torch

from configuration.config import *


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            # self.recent_epoch = epoch
            # state_dict = kwargs['state_dict']
            # state_dict['epoch'] = kwargs['epoch']
            # state_dict[ACCUM_ITER_DICT_KEY] = kwargs[ACCUM_ITER_DICT_KEY]
            save_state_dict(kwargs, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        # state_dict = kwargs['state_dict']
        # state_dict['epoch'] = kwargs['epoch']
        # state_dict[ACCUM_ITER_DICT_KEY] = kwargs[ACCUM_ITER_DICT_KEY]
        save_state_dict(kwargs, self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.best_metric = -99999.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            # print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            logging.info("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric

            # state_dict = kwargs['state_dict']
            # state_dict['epoch'] = kwargs['epoch']
            # state_dict[ACCUM_ITER_DICT_KEY] = kwargs[ACCUM_ITER_DICT_KEY]
            save_state_dict(kwargs, self.checkpoint_path, self.filename)


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs[ACCUM_ITER_DICT_KEY])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs[ACCUM_ITER_DICT_KEY])

    def complete(self, *args, **kwargs):
        self.writer.close()

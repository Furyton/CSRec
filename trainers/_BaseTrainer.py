from abc import *

from models.base import BaseModel

class _AbstractBaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_epochs = args.num_epochs
        self.batch_size = args.train_batch_size
        self.num_items = args.num_items
        self.enable_neg_sample = args.test_negative_sample_size != 0
        self.log_period_as_iter = args.log_period_as_iter
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _train_one_epoch(self):
        pass

    @abstractmethod
    def close_training(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def test(self, export_root: str):
        pass

    @abstractmethod
    def test_with_given_state_path(self, state_path):
        pass

    @abstractmethod
    def _create_log_data(self, metrics: dict = None):
        pass
    
    # @abstractmethod
    # def _create_state_dict(self):
    #     pass

from copy import deepcopy
import logging
from models.DeepFM import DeepFM
from models.base import BaseModel

from models.bert import BERTModel
from models.Caser import CaserModel
from models.GRU4rec import GRU4RecModel
from models.pop import POPModel
from models.NextItNet import NextItNet

MODELS = {
    BERTModel.code(): BERTModel,
    POPModel.code(): POPModel,
    GRU4RecModel.code(): GRU4RecModel,
    CaserModel.code(): CaserModel,
    DeepFM.code(): DeepFM,
    NextItNet.code(): NextItNet,
}

def model_factory(args, model_type: str, dataset: list) -> BaseModel:
    if model_type.lower() not in MODELS.keys():
        logging.fatal(f"{model_type} has not been implemented yet.")
        raise NotImplementedError(f"{model_type} has not been implemented yet.")
    
    from configuration.utils import load_model_config

    model_args = load_model_config(model_code=model_type)

    model = MODELS[model_type]

    return model(model_args, deepcopy(dataset), args.device, args.max_len)

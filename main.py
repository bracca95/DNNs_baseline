import os
import sys
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from src.config_parser import Config
from src.datasets import DefectViews
from src.model import MLP
from src.tools import Logger
from src.train import Trainer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# TODO:
# filt for dataset in config
# batch size in config
# epochs in config

if __name__=="__main__":
    config = Config.deserialize("config/config.json")
    
    if config.crop_size is None:
        raise ValueError("define crop size in config.json")

    dataset = DefectViews(config.dataset_path, config.crop_size, filt=["bubble", "point"])
    if config.dataset_mean is None and config.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        DefectViews.compute_mean_std(dataset, config)
        sys.exit(0)

    train_test_split = int(len(dataset)*0.8)
    trainset, testset = random_split(dataset, [train_test_split, len(dataset) - train_test_split])
    
    in_dim = config.crop_size if config.image_size is None else config.image_size
    model = MLP(in_dim * in_dim, len(config.defect_class))

    trainer = Trainer(trainset, model)
    trainer.train(config)

    Logger.instance().debug("program terminated")
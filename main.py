import sys
import torch
import random
import numpy as np

from torch.utils.data import random_split
from src.config_parser import Config
from src.datasets import DefectViews, MNIST, BubblePoint
from src.models import MLP, CNN, ResCNN
from src.tools import Logger
from src.train import Trainer
from src.test import Tester

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    try:
        config = Config.deserialize("config/config.json")
    except Exception as e:
        Logger.instance().error(e.args)
        sys.exit(-1)
    
    if config.crop_size is None and config.dataset != "mnist":
        raise ValueError("define crop size in config.json")

    # load desired dataset
    if config.dataset == "mnist":
        dataset = MNIST(config.dataset_path, config.crop_size, config.image_size)
    elif config.dataset == "all":
        dataset = DefectViews(config.dataset_path, config.crop_size, img_size=config.image_size)
    else:
        dataset = BubblePoint(config.dataset_path, config.crop_size, img_size=config.image_size)
    
    # compute mean and variance of the dataset if not done yet
    if config.dataset != "mnist" and config.dataset_mean is None and config.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        DefectViews.compute_mean_std(dataset, config)
        sys.exit(0)

    if type(dataset) is MNIST:
        trainset = dataset.get_train_dataset()
        testset = dataset.get_test_dataset()
    else:
        train_test_split = int(len(dataset)*0.8)
        trainset, testset = random_split(dataset, [train_test_split, len(dataset) - train_test_split])

    if config.mode == "mlp":
        Logger.instance().debug("running MLP")
        model = MLP(dataset.in_dim * dataset.in_dim, dataset.out_dim)
    elif config.mode == "rescnn":
        Logger.instance().debug("running ResCNN")
        model = ResCNN(dataset.in_dim, dataset.out_dim)
    elif config.mode == "cnn":
        Logger.instance().debug("running CNN")
        model = CNN(dataset.out_dim)
    else:
        raise ValueError("either 'mlp' or 'cnn' or 'rescnn'")

    if config.train:
        Logger.instance().debug("Starting training...")
        trainer = Trainer(trainset, model)
        trainer.train(config)
    else:
        Logger.instance().debug("Starting testing...")
        tester = Tester(testset, model, "checkpoints/model.pt")
        tester.test(config)

    Logger.instance().debug("program terminated")
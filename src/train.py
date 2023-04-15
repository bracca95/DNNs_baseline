from torch.utils.data import Dataset, DataLoader

from src.config_parser import Config


class Trainer:

    @staticmethod
    def train(trainset: Dataset, config: Config):
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

        for idx, (img, label) in enumerate(trainloader):
            print()
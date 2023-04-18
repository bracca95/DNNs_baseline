import os
import torch
import torch.optim as optim

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.model import MLP
from src.config_parser import Config
from src.tools import Logger, TBWriter


class Trainer:

    def __init__(self, trainset: Dataset, model: MLP):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.trainset = trainset
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        # tensorboard
        self.writer = TBWriter.instance().get_writer()

        Logger.instance().debug(f"device: {self.device.type}")

    @staticmethod
    def compute_accuracy(y_pred: torch.Tensor, y: torch.Tensor):
        top_pred = y_pred.argmax(1, keepdim=True)           # select the max class (the one with the highest score)
        correct = top_pred.eq(y.view_as(top_pred)).sum()    # count the number of correct predictions
        return correct.float() / y.shape[0]                 # compute percentage of correct predictions (accuracy score)
    
    def train(self, config: Config):
        self.model.train()
        trainloader = DataLoader(self.trainset, batch_size=config.batch_size, shuffle=True)
        
        # tensorboard
        example_data, examples_target = next(iter(trainloader))
        shape = config.crop_size if config.image_size is None else config.image_size
        self.writer.add_graph(self.model, example_data.to(self.device).reshape(-1, shape * shape))
        self.writer.close()

        model_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        best_loss = float('inf')
        for epoch in range(100):
            epoch_loss = 0
            epoch_acc = 0

            for (image, label) in tqdm(trainloader, desc="Training", leave=False):
                image = image.to(self.device)
                label = label.to(self.device)

                # forward pass
                pred = self.model(image)                        # [batch_size, n_classes]
                loss = self.criterion(pred, label)
                acc = Trainer.compute_accuracy(pred.data, label)

                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            epoch_loss = epoch_loss / len(trainloader)
            epoch_acc = epoch_acc / len(trainloader)

            if epoch_loss < best_loss and epoch > 0:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pt"))
                Logger.instance().debug(f"saving model at iteration {epoch}. Find it at: {model_dir}")

            # print(f"Epoch: {epoch}, Train Loss: {epoch_loss:.3f}")
            self.writer.add_scalar("Loss", epoch_loss, epoch)
            self.writer.add_scalar("Accuracy", epoch_acc, epoch)

        self.writer.close()
        
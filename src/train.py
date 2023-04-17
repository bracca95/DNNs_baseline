import os
import torch
import torch.optim as optim

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.model import MLP
from src.config_parser import Config


class Trainer:

    def __init__(self, trainset: Dataset, model: MLP):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.trainset = trainset
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

    @staticmethod
    def calculate_accuracy(y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
    
    def train(self, config: Config):
        trainloader = DataLoader(self.trainset, batch_size=config.batch_size, shuffle=True)
        best_loss = float('inf')
        model_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        for epoch in range(100):
            epoch_loss = 0
            epoch_acc = 0

            self.model.train()

            for (image, label) in tqdm(trainloader, desc="Training", leave=False):
                self.optimizer.zero_grad()

                image = image.to(self.device)
                label = label.to(self.device)

                pred = self.model(image)
                loss = self.criterion(pred, label)
                acc = Trainer.calculate_accuracy(pred, label)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            epoch_loss = epoch_loss / len(trainloader)
            epoch_acc = epoch_acc / len(trainloader)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pt"))

            print(f"Epoch: {epoch}, Train Loss: {epoch_loss:.3f}")
        
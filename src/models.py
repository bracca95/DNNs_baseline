# https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
# https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md

from torch import nn


def conv3x3(in_channels: int, out_channels: int, ksize: int, stride: int, pad: int):
    return nn.Conv2d(in_channels, out_channels, ksize, stride, pad)


class BasicBlock(nn.Module):
    """BasicBlock is a residual block (skip connection)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, ksize=3, stride=1, pad=1)
        self.conv2 = conv3x3(out_channels, out_channels, ksize=3, stride=1, pad=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        # https://paperswithcode.com/method/residual-block
        residual = x

        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm(out)

        # align number of channels
        if residual.shape[1] != 1 and residual.shape[1] != out.shape[1]:
            conv1x1 = nn.Conv2d(residual.shape[1], out.shape[1], kernel_size=1, stride=1).to(residual.device)
            residual = conv1x1(residual)
        
        out += residual

        return self.activation(out)


class CNN(nn.Module):

    def __init__(self, out_feat: int):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, out_feat),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)

        out = out.view(out.size(0), -1)
        return self.fc1(out)


class ResCNN(nn.Module):

    def __init__(self, in_size: int, out_feat: int):
        super().__init__()

        self.seq1 = BasicBlock(1, 4)
        self.seq2 = BasicBlock(4, 32)
        self.seq3 = BasicBlock(32, 64)
        self.linear = nn.Sequential(
            nn.Linear(in_size * in_size * 64, out_feat),
            nn.Softmax()
        ) 

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)
        out = self.seq3(out)
        
        out = out.view(out.size(0), -1)
        return self.linear(out)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        ratio = in_dim // 4

        self.hidden1 = nn.Sequential(
            nn.Linear(in_dim, 3*ratio),
            nn.BatchNorm1d(3*ratio),
            nn.ReLU(),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(3*ratio, 2*ratio),
            nn.BatchNorm1d(2*ratio),
            nn.ReLU(),
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(2*ratio, ratio),
            nn.BatchNorm1d(ratio),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(ratio, out_dim),
            nn.Softmax(),
        )
            

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.out(x)
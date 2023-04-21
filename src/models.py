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
        # check both in_dim/out_dim and if out_dim is the same as len(config.defect_class)

        self.hidden1 = nn.Sequential(
            nn.Linear(in_dim, 2600),
            nn.BatchNorm1d(2600),
            nn.ReLU(),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(2600, 1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(1600, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.Softmax(),
        )
            

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.out(x)
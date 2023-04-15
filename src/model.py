# https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
# https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
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
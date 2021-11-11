import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mlp']


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 150)
        self.fc3 = nn.Linear(150, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        self.x = x
        self.flat_out = self.x.view(-1, 784)
        self.fc1_out = self.fc1(self.flat_out)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out)
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out)
        return self.fc3_out


def mlp(**kwargs):
    return MLP(**kwargs)

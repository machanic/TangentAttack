import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet']


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(1024, 64)
        # self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

        for k in ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']:
            w = self.__getattr__(k)
            torch.nn.init.kaiming_normal_(w.weight.data)
            w.bias.data.fill_(0)

        self.out = dict()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.drop1(x)

        x = self.fc2(x)

        return x


def lenet(**kwargs):
    return lenet(**kwargs)

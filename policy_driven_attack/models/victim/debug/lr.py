import torch
import torch.nn as nn

__all__ = ['lr']


class LR(nn.Module):
    def __init__(self, num_classes=2):
        super(LR, self).__init__()
        self.fc = nn.Linear(2, num_classes)
        self.fc.weight.data[:] = torch.FloatTensor([[0., -1.], [0., 1.]])
        self.fc.bias.data[:] = torch.FloatTensor([0.5, -0.5])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def lr(**kwargs):
    return LR(**kwargs)

import torch.nn as nn

__all__ = ['lr']


class LR(nn.Module):
    def __init__(self, num_classes=10):
        super(LR, self).__init__()
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def lr(**kwargs):
    return LR(**kwargs)

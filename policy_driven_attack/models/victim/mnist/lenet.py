import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet', 'lenet150']


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        self.x = x
        self.conv1_out = self.conv1(self.x)
        self.pool1_out, self.pool1_ind = F.max_pool2d(self.conv1_out, (2, 2), return_indices=True)
        self.conv2_out = self.conv2(self.pool1_out)
        self.pool2_out, self.pool2_ind = F.max_pool2d(self.conv2_out, (2, 2), return_indices=True)
        self.flat_out = self.pool2_out.view(-1, self.num_flat_features(self.pool2_out))
        self.fc1_out = self.fc1(self.flat_out)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out)

        return self.fc2_out

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet150(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet150, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 4 * 4, 150)
        self.fc2 = nn.Linear(150, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        self.x = x
        self.conv1_out = self.conv1(self.x)
        self.pool1_out, self.pool1_ind = F.max_pool2d(self.conv1_out, (2, 2), return_indices=True)
        self.conv2_out = self.conv2(self.pool1_out)
        self.pool2_out, self.pool2_ind = F.max_pool2d(self.conv2_out, (2, 2), return_indices=True)
        self.flat_out = self.pool2_out.view(-1, self.num_flat_features(self.pool2_out))
        self.fc1_out = self.fc1(self.flat_out)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out)

        return self.fc2_out

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def lenet(**kwargs):
    return LeNet(**kwargs)


def lenet150(**kwargs):
    return LeNet150(**kwargs)


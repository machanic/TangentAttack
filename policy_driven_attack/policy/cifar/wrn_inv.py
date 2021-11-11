import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.common import normalization, inv_forward

__all__ = ['wrn_28_10_drop_inv']


def normalization16(normalization_type, planes):
    return normalization(normalization_type, planes, group_size=16)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, normalization_type='none'):
        super(BasicBlock, self).__init__()
        self.bn1 = normalization16(normalization_type, in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normalization16(normalization_type, out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, normalization_type='none'):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, normalization_type)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, normalization_type='none'):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate,
                                normalization_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WRNInv(nn.Module):
    def __init__(self, depth, widen_factor=1, drop_rate=0.0, use_tanh=True, calibrate=True,
                 normalization_type='none', input_size=32, init_std=1 / 32.):
        super(WRNInv, self).__init__()
        # save arguments
        self.depth = depth
        self.input_size = input_size
        self.init_std = init_std
        self.normalization_type = normalization_type
        self.use_tanh = use_tanh
        self.calibrate = calibrate

        # main body of wrn inverse
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = normalization16(normalization_type, nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], 10)
        self.nChannels = nChannels[3]

        # mean / std output
        self.normal_mean = nn.Parameter(torch.zeros(3, input_size, input_size))
        self.normal_logstd = nn.Parameter(torch.ones(1) * math.log(self.init_std))
        self.mean_shape = (3, input_size, input_size)

        # whiten function placeholder
        # we can not assign value to it now since input_mean / input_std is unknown yet
        # we will assign appropriate function in StandardPolicyModel.__init__
        self.whiten_func = lambda t: 1 / 0

        # coefficient to tune between prior knowledge and empty network
        self.empty_coeff = nn.Parameter(torch.ones(1) * 0.5)
        self.empty_normal_mean = nn.Parameter(torch.zeros(3, input_size, input_size))

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_logit(self, x):
        # whiten input
        x = self.whiten_func(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def forward(self, adv_image, image, label, target, output_fields):
        output = inv_forward(
            adv_image=adv_image, image=image, label=label, target=target, get_logit=self.get_logit,
            normal_mean=self.normal_mean, empty_coeff=self.empty_coeff, empty_normal_mean=self.empty_normal_mean,
            training=self.training, calibrate=self.calibrate, output_fields=output_fields)
        if 'std' in output_fields:
            output['std'] = self.normal_logstd.exp()
        return output

    def rescale(self, scale):
        raise NotImplementedError


def wrn_28_10_drop_inv(**kwargs):
    return WRNInv(depth=28, widen_factor=10, drop_rate=0.3, **kwargs)

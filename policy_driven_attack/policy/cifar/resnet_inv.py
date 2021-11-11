import math
import torch
import torch.nn as nn
from policy.common import normalization, inv_forward

__all__ = ['resnet20_inv', 'resnet32_inv']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def normalization16(normalization_type, planes):
    return normalization(normalization_type, planes, group_size=16)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalization_type='none'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization16(normalization_type, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization16(normalization_type, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalization_type='none'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization16(normalization_type, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normalization16(normalization_type, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normalization16(normalization_type, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetInv(nn.Module):
    def __init__(self, depth, use_tanh=True, calibrate=True, normalization_type='none', input_size=32, init_std=1/32.):
        super(ResNetInv, self).__init__()
        # save arguments
        self.depth = depth
        self.input_size = input_size
        self.init_std = init_std
        self.normalization_type = normalization_type
        self.use_tanh = use_tanh
        self.calibrate = calibrate

        # main body of resnet inverse
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = normalization16(normalization_type, 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, 10)

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if normalization_type in ['bn', 'gn']:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normalization16(self.normalization_type, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, normalization_type=self.normalization_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normalization_type=self.normalization_type))

        return nn.Sequential(*layers)

    def get_logit(self, x):
        # whiten input
        x = self.whiten_func(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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


def resnet20_inv(**kwargs):
    return ResNetInv(depth=20, **kwargs)


def resnet32_inv(**kwargs):
    return ResNetInv(depth=32, **kwargs)

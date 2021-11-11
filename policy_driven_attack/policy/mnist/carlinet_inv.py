import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.common import inv_forward

__all__ = ['carlinet_inv']


class CarliNetInv(nn.Module):
    def __init__(self, use_tanh=True, calibrate=True, input_size=28, init_std=1/28.):
        super(CarliNetInv, self).__init__()
        # save arguments
        self.input_size = input_size
        self.init_std = init_std
        self.use_tanh = use_tanh
        self.calibrate = calibrate

        # main body of carlinet inverse
        self.conv2d_1 = nn.Conv2d(1, 32, 3)
        self.conv2d_2 = nn.Conv2d(32, 32, 3)
        self.conv2d_3 = nn.Conv2d(32, 64, 3)
        self.conv2d_4 = nn.Conv2d(64, 64, 3)

        self.dense_1 = nn.Linear(1024, 200)
        self.dense_2 = nn.Linear(200, 200)
        self.dense_3 = nn.Linear(200, 10)

        # mean / std output
        self.normal_mean = nn.Parameter(torch.zeros(1, input_size, input_size))
        self.normal_logstd = nn.Parameter(torch.ones(1) * math.log(self.init_std))
        self.mean_shape = (1, input_size, input_size)

        # whiten function placeholder
        # we can not assign value to it now since input_mean / input_std is unknown yet
        # we will assign appropriate function in StandardPolicyModel.__init__
        self.whiten_func = lambda t: 1 / 0

        # coefficient to tune between prior knowledge and empty network
        self.empty_coeff = nn.Parameter(torch.ones(1) * 0.5)
        self.empty_normal_mean = nn.Parameter(torch.zeros(1, input_size, input_size))

        # init weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_logit(self, x):
        # whiten input
        x = self.whiten_func(x)

        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2d_3(x)
        x = F.relu(x)
        x = self.conv2d_4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        # carlini's keras model data format: (N, H, W, C)
        # pytorch data format: (N, C, H, W)
        # we need to transpose pytorch data format into keras data format, to make sure the flatten operator
        # has the same effect.
        x = x.transpose(1, 2).transpose(2, 3).contiguous().view(x.shape[0], -1)
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dense_2(x)
        if self.use_tanh:
            x = F.tanh(x)
        else:
            x = F.relu(x)
        logit = self.dense_3(x)

        return logit

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


def carlinet_inv(**kwargs):
    return CarliNetInv(**kwargs)

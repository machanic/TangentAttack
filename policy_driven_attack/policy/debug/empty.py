import math
import torch
import torch.nn as nn

__all__ = ['empty']


class Empty(nn.Module):
    def __init__(self, init_std=0.001):
        super(Empty, self).__init__()
        # save arguments
        self.init_std = init_std

        # mean / std output
        self.normal_mean = nn.Parameter(torch.zeros(2))
        self.normal_logstd = nn.Parameter(torch.ones(1) * math.log(self.init_std))
        self.mean_shape = (2,)

        # whiten function placeholder
        # we can not assign value to it now since input_mean / input_std is unknown yet
        # we will assign appropriate function in StandardPolicyModel.__init__
        self.whiten_func = lambda t: 1 / 0

    def forward(self, adv_image, image, label, target, output_fields):
        output = dict()
        if 'grad' in output_fields:
            output['grad'] = self.normal_mean.view(1, *self.mean_shape).repeat(adv_image.shape[0], 1)
        if 'std' in output_fields:
            output['std'] = self.normal_logstd.exp()
        return output

    def rescale(self, scale):
        self.normal_mean.data[:] *= scale


def empty(**kwargs):
    return Empty(**kwargs)

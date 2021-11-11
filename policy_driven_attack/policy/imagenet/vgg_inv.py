import math
import torch
import torch.nn as nn
from policy.common import normalization, inv_forward

__all__ = ['vgg11_inv', 'vgg13_inv', 'vgg16_inv', 'vgg19_inv']


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGInv(nn.Module):
    def __init__(self, vgg_name, use_tanh=True, calibrate=True,
                 normalization_type='none', input_size=224, init_std=1/224.):
        super(VGGInv, self).__init__()
        # save arguments
        self.input_size = input_size
        self.init_std = init_std
        self.normalization_type = normalization_type
        self.use_tanh = use_tanh
        self.calibrate = calibrate

        # main body of vgg inverse
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
        )

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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for layer_index, x in enumerate(cfg):
            if x == 'M':
                # append max pooling layer
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # append conv->(bn/gn)->activation layer
                # determine activation function: we use tanh for last layer if self.use_tanh is set to True
                is_last_layer = True
                for xx in cfg[layer_index+1:]:
                    if xx != 'M':
                        is_last_layer = False
                if self.use_tanh and is_last_layer:
                    activation = nn.Tanh()
                else:
                    activation = nn.ReLU(inplace=True)

                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           normalization(self.normalization_type, x),
                           activation]
                in_channels = x
        # imagenet vgg13 does not do that
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_logit(self, x):
        # whiten input
        x = self.whiten_func(x)

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

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


def vgg11_inv(**kwargs):
    return VGGInv('vgg11', **kwargs)


def vgg13_inv(**kwargs):
    return VGGInv('vgg13', **kwargs)


def vgg16_inv(**kwargs):
    return VGGInv('vgg16', **kwargs)


def vgg19_inv(**kwargs):
    return VGGInv('vgg19', **kwargs)

# code borrowed from https://github.com/milesial/Pytorch-UNet
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['unet']


def normalization(normalization_type, planes):
    if normalization_type == 'none':
        return nn.Identity()
    elif normalization_type == 'bn':
        return nn.BatchNorm2d(planes)
    elif normalization_type == 'gn':
        return nn.GroupNorm(num_groups=planes//32, num_channels=planes)
    else:
        raise ValueError('Unknown normalization method: {}'.format(normalization))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normalization_type='none'):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = list()
        self.double_conv.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1))
        self.double_conv.append(normalization(normalization_type, mid_channels))
        self.double_conv.append(nn.ReLU(inplace=True))
        self.double_conv.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1))
        self.double_conv.append(normalization(normalization_type, out_channels))
        self.double_conv.append(nn.ReLU(inplace=True))

        # make double_conv as a nn.Module with nn.Sequential
        self.double_conv = nn.Sequential(*self.double_conv)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalization_type='none'):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization_type=normalization_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, normalization_type='none'):
        super(Up, self).__init__()

        # if not bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2,
                                   normalization_type=normalization_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, normalization_type=normalization_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_size=28, n_channels=1, bilinear=True,
                 normalization_type='none', base_width=16, init_std=1/28.):
        super(UNet, self).__init__()
        # save arguments
        self.input_size = input_size
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.normalization_type = normalization_type
        self.base_width = base_width
        self.init_std = init_std

        # main body of unet
        widths = [base_width, base_width * 2, base_width * 4]
        self.inc = DoubleConv(n_channels, widths[0], normalization_type=normalization_type)
        self.down1 = Down(widths[0], widths[1], normalization_type=normalization_type)
        factor = 2 if bilinear else 1
        self.down2 = Down(widths[1], widths[2] // factor, normalization_type=normalization_type)
        self.up1 = Up(widths[2], widths[1] // factor, bilinear, normalization_type=normalization_type)
        self.up2 = Up(widths[1], widths[0], bilinear, normalization_type=normalization_type)
        self.outc = OutConv(widths[0], n_channels)

        # mean / std output
        self.normal_mean = nn.Parameter(torch.zeros(1, input_size, input_size))
        self.normal_logstd = nn.Parameter(torch.ones(1) * math.log(self.init_std))
        self.mean_shape = (1, input_size, input_size)

        # init weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, adv_image, image, label, target, output_fields):
        output = dict()
        if 'grad' in output_fields:
            # whiten input
            adv_image = self.whiten_func(adv_image)

            x = adv_image
            x1 = self.inc(x)      # widths[0] channels
            x2 = self.down1(x1)   # widths[1] channels
            x3 = self.down2(x2)   # widths[2] // 2 channels if bilinear, else widths[2]
            x = self.up1(x3, x2)  # widths[1] // 2 channels if bilinear, else widths[1]
            x = self.up2(x, x1)   # widths[0] channels
            x = self.outc(x)      # n_channels, same as input image
            x = x + self.normal_mean.view(1, *self.mean_shape).repeat(x.shape[0], 1, 1, 1)

            output['grad'] = x

        if 'std' in output_fields:
            output['std'] = self.normal_logstd.exp()
        return output

    def rescale(self, scale):
        self.outc.conv.weight.data[:] *= scale
        if self.outc.conv.bias:
            self.outc.conv.bias.data[:] *= scale
        self.normal_mean.data[:] *= scale


def unet(**kwargs):
    return UNet(**kwargs)

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data.dataset import Dataset
import os
import torchvision.models as models
import numpy as np



def proj(pert, eps, norm, discrete):
    # project image into epsilon ball (either L_inf norm or L_2 norm)
    # if discrete=True, project into the boundary of the ball instead of the ball
    if norm == "linf":
        if discrete:
            return eps * pert.sign()
        else:
            return pert.clamp(-eps, eps)
    else:
        pert_norm = torch.norm(pert.view(pert.shape[0], -1), dim=1)
        pert_norm = torch.where(pert_norm > eps, pert_norm / eps,
                                torch.ones_like(pert_norm))
        return pert / pert_norm.view(-1, 1, 1, 1)


def latent_proj(pert, eps):
    # project the latent variables (i.e., FFT variables)
    # into the epsilon L_2 ball
    pert_norm = torch.norm(pert, dim=1) / eps
    return pert.div_(pert_norm.view(-1, 1))


def fft_transform(pert, image_height, image_width,channels, dim):
    # single channel fft transform
    res = torch.zeros(pert.shape[0], channels, image_height, image_width)
    for i in range(pert.shape[0]):
        t = torch.zeros((image_height, image_width, 2))
        t[:dim, :dim] = pert[i].view(dim, dim, 2)
        res[i, 0] = torch.irfft(t, 2, normalized=True, onesided=False)
    return res.cuda()


def fft_transform_mc(pert, image_height, image_width, channels):
    # multi-channel FFT transform (each channel done separately)
    # performs a low frequency perturbation (set of allowable frequencies determined by shape of pert)
    res = torch.zeros(pert.shape[0], channels, image_height, image_width)
    t_dim = int(np.sqrt(pert.shape[1] / (2 * channels)))
    for i in range(pert.shape[0]):
        t = torch.zeros((channels, image_height, image_width, 2))
        t[:, :t_dim, :t_dim, :] = pert[i].view(channels, t_dim, t_dim, 2)
        res[i] = torch.irfft(t, channels, normalized=True, onesided=False)
    return res


def transform(pert, dataset_name, image_height, image_width, channels, dim):
    if dataset_name == 'MNIST':
        return fft_transform(pert, image_height, image_width, channels, dim)
    else:
        return fft_transform_mc(pert, image_height, image_width, channels)





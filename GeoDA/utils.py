import numpy as np
import os
import torch
import copy
from math import cos, sqrt, pi
from numba import jit


def dct(x, y, v, u, n):
    # Normalisation
    def alpha(a):
        if a == 0:
            return sqrt(1.0 / n)
        else:
            return sqrt(2.0 / n)

    return alpha(u) * alpha(v) * cos(((2 * x + 1) * (u * pi)) / (2 * n)) * cos(((2 * y + 1) * (v * pi)) / (2 * n))

@jit
def generate_2d_dct_basis(root_path, image_height, sub_dim=75):
    path = "{}/attacked_images/GeoDA/2d_dct_basis_height_{}_subdim_{}.npy".format(root_path, image_height, sub_dim)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    if os.path.exists(path):
        return np.load(path)
    n = image_height  # Assume square image, so we don't have different xres and yres

    # We can get different frequencies by setting u and v
    # Here, we have a max u and v to loop over and display
    # Feel free to adjust
    maxU = sub_dim
    maxV = sub_dim

    dct_basis = []
    for u in range(0, maxU):
        for v in range(0, maxV):
            basisImg = np.zeros((n, n))
            for y in range(0, n):
                for x in range(0, n):
                    basisImg[y, x] = dct(x, y, v, u, max(n, maxV))
            dct_basis.append(basisImg)
    dct_basis = np.mat(np.reshape(dct_basis, (maxV*maxU, n*n))).transpose()
    np.save(path, dct_basis)
    return dct_basis


def clip_image_values(x, minv, maxv):
    if not isinstance(minv, torch.Tensor):
        return torch.clamp(x,min=minv,max=maxv)
    return torch.min(torch.max(x, minv), maxv)


def valid_bounds(img, delta=255):

    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.int)

    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)

    # Compute the bounds
    lb = im - delta
    ub = im + delta

    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))

    # Change types to uint8
    lb = lb.astype(np.uint8)
    ub = ub.astype(np.uint8)

    return lb, ub


def inv_tf(x, mean, std):

    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x


def inv_tf_pert(r):

    pert = np.sum(np.absolute(r), axis=0)
    pert[pert != 0] = 1

    return pert


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '

    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))

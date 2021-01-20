from typing import Callable, Union, Optional, Tuple, List, Any, Dict
from scipy import fft
import numpy as np
import torch
from typing import TypeVar
T = TypeVar("T")
###########################
# DCT Functions
###########################

def atleast_kd(x: torch.Tensor, k: int) -> torch.Tensor:
    shape = x.size() + (1,) * (k - x.dim())
    return x.view(shape)

def dct2(a: Any) -> Any:
    return fft.dct(fft.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a: Any) -> Any:
    return fft.idct(fft.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def dct2_8_8(image: Any, mask: Any = None) -> Any:
    if mask is None:
        mask = np.ones((8, 8))
    if mask.shape != (8, 8):
        raise ValueError("Mask have to be with a size of (8, 8)")

    imsize = image.shape
    dct = np.zeros(imsize)

    for channel in range(imsize[0]):
        for i in np.r_[:imsize[1]:8]:
            for j in np.r_[:imsize[2]:8]:
                dct_i_j = dct2(image[channel, i:(i + 8), j:(j + 8)])
                dct[channel, i:(i + 8), j:(j + 8)] = dct_i_j * mask[:dct_i_j.shape[0], :dct_i_j.shape[1]]
    return dct


def idct2_8_8(dct: Any) -> Any:
    im_dct = np.zeros(dct.shape)

    for channel in range(dct.shape[0]):
        for i in np.r_[:dct.shape[1]:8]:
            for j in np.r_[:dct.shape[2]:8]:
                im_dct[channel, i:(i + 8), j:(j + 8)] = idct2(dct[channel, i:(i + 8), j:(j + 8)])
    return im_dct


def dct2_full(image: Any, mask: Any = None) -> Any:
    if mask is None:
        mask = np.ones(image.shape[-2:])

    imsize = image.shape
    dct = np.zeros(imsize)

    for channel in range(imsize[0]):
        dct_i_j = dct2(image[channel])
        dct[channel] = dct_i_j * mask
    return dct


def idct2_full(dct: Any) -> Any:
    im_dct = np.zeros(dct.shape)

    for channel in range(dct.shape[0]):
        im_dct[channel] = idct2(dct[channel])
    return im_dct


def get_zig_zag_mask(frequence_range: Tuple[float, float], mask_shape: Tuple[int, int] = (8, 8)) -> Any:
    mask = np.zeros(mask_shape)
    s = 0
    total_component = sum(mask.flatten().shape)

    if frequence_range[1] <= 1:
        n_coeff = int(total_component * frequence_range[1])
    else:
        n_coeff = int(frequence_range[1])

    if frequence_range[0] <= 1:
        min_coeff = int(total_component * frequence_range[0])
    else:
        min_coeff = int(frequence_range[0])

    while n_coeff > 0:
        for i in range(min(s + 1, mask_shape[0])):
            for j in range(min(s + 1, mask_shape[1])):
                if i + j == s:
                    if min_coeff > 0:
                        min_coeff -= 1
                        continue

                    if s % 2:
                        mask[i, j] = 1
                    else:
                        mask[j, i] = 1
                    n_coeff -= 1
                    if n_coeff == 0:
                        return mask
        s += 1
    return mask

class LpDistance(object):
    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def __call__(self, references: T, perturbed: T) -> T:
        """Calculates the distances from references to perturbed using the Lp norm.
        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)

    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        """Clips the perturbations to epsilon and returns the new perturbed
        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
        Returns:
            A tenosr like perturbed but with the perturbation clipped to epsilon.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        p = y - x
        if self.p == ep.inf:
            clipped_perturbation = ep.clip(p, -epsilon, epsilon)
            return restore_type(x + clipped_perturbation)
        norms = ep.norms.lp(flatten(p), self.p, axis=-1)
        norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
        factor = epsilon / norms
        factor = ep.minimum(1, factor)  # clipping -> decreasing but not increasing
        if self.p == 0:
            if (factor == 1).all():
                return perturbed
            raise NotImplementedError("reducing L0 norms not yet supported")
        factor = atleast_kd(factor, x.ndim)
        clipped_perturbation = factor * p
        return restore_type(x + clipped_perturbation)


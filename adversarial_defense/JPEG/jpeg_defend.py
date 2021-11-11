from PIL import Image
import numpy as np
from scipy.misc import imread
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
import torch

class JPEG:
    def __init__(self, quality=75):

        self.predictions_is_correct = False
        self.use_larger_step_size = True
        self.use_smoothed_grad = True

        # For dataprior attacks. gamma = A^2 * D / d in the paper
        self.gamma = 4.0
        self.quality = quality

    def forward(self, imgs):
        images = imgs.detach().cpu().numpy().copy()
        if len(images.shape) == 3:
            images = images[np.newaxis]
        for i in range(images.shape[0]):
            img = Image.fromarray((images[i] * 255.0).astype(np.uint8), 'RGB')
            virtualpath = BytesIO()
            img.save(virtualpath, "JPEG", quality=self.quality)
            images[i] = imread(virtualpath).astype(np.float) / 255.0

        return torch.from_numpy(images).cuda()
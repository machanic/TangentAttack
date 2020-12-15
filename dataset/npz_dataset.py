from torch.utils import data
import torch

from config import PY_ROOT
import numpy as np

class NpzDataset(data.Dataset):
    def __init__(self, dataset):
        file_path = "{}/attacked_images/{}/{}_images.npz".format(PY_ROOT, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        self.images = file_data["images"]
        self.labels = file_data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return torch.from_numpy(image),label


class NpzExtraDataset(NpzDataset):
    def __init__(self, dataset):
        super(NpzExtraDataset, self).__init__(dataset)
        file_path = "{}/attacked_images/{}/{}_images_for_candidate.npz".format(PY_ROOT, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        self.images = file_data["images"]
        self.labels = file_data["labels"]


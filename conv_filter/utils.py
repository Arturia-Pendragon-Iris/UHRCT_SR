import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from .filter import perform_filter, normalize_ct


try:
    from monai.transforms import Compose, RandAffine, RandFlip, RandRotate90
except ImportError:
    Compose = None


def build_train_transforms(spatial_size=(512, 512)):
    if Compose is None:
        return None
    return Compose(
        [
            RandAffine(
                prob=0.5,
                padding_mode="zeros",
                spatial_size=spatial_size,
                translate_range=(64, 64),
                rotate_range=(np.pi / 8, np.pi / 8),
                scale_range=(-0.5, 0.5),
            ),
            RandFlip(prob=0.5),
            RandRotate90(prob=0.5),
        ]
    )


def refine_ct(ct_array, hu_min=-1000.0, hu_max=600.0):
    return normalize_ct(ct_array, hu_min=hu_min, hu_max=hu_max)


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device=None, transform=True, hu_min=-1000.0, hu_max=600.0):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.device = device
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.transforms = build_train_transforms() if transform else None

        file_list = []
        for root, _, files in os.walk(dataset_dir):
            for name in files:
                if name.endswith(".npz"):
                    file_list.append(os.path.join(root, name))
        self.file_list = sorted(file_list)

    def __getitem__(self, index):
        raw_array = np.load(self.file_list[index])["arr_0"]
        if raw_array.ndim == 3:
            raw_array = raw_array[0]

        image = refine_ct(raw_array, hu_min=self.hu_min, hu_max=self.hu_max)
        image = image[np.newaxis]
        if self.transforms is not None:
            image = np.clip(self.transforms(image), 0.0, 1.0)

        structure = perform_filter(
            image[0],
            hu_min=0.0,
            hu_max=1.0,
        )[np.newaxis]

        raw_tensor = torch.tensor(image).clone().to(torch.float)
        filtered_tensor = torch.tensor(structure).clone().to(torch.float)
        if self.device is not None:
            raw_tensor = raw_tensor.to(self.device)
            filtered_tensor = filtered_tensor.to(self.device)
        return raw_tensor, filtered_tensor

    def __len__(self):
        return len(self.file_list)

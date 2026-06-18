import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def normalize_ct(ct_array, hu_min=-1000.0, hu_max=600.0):
    ct_array = np.asarray(ct_array, dtype=np.float32)
    return np.clip((ct_array - hu_min) / (hu_max - hu_min), 0.0, 1.0)


def _list_npz(root):
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith(".npz"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def _read_npz_slice(path, random_slice=True):
    array = np.load(path)["arr_0"]
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        index = random.randrange(array.shape[0]) if random_slice else array.shape[0] // 2
        return array[index]
    raise ValueError("Unsupported npz shape {} in {}".format(array.shape, path))


class CTNpzDataset(Dataset):
    def __init__(
        self,
        hr_dir,
        lr_dir=None,
        crop_size=256,
        scale=4,
        hu_min=-1000.0,
        hu_max=600.0,
        random_crop=True,
        random_flip=True,
    ):
        super().__init__()
        self.hr_files = _list_npz(hr_dir)
        if not self.hr_files:
            raise ValueError("No .npz files found in {}".format(hr_dir))
        self.lr_files = None
        if lr_dir is not None:
            lr_files = {os.path.basename(path): path for path in _list_npz(lr_dir)}
            self.lr_files = [lr_files.get(os.path.basename(path)) for path in self.hr_files]
            missing = [os.path.basename(self.hr_files[i]) for i, path in enumerate(self.lr_files) if path is None]
            if missing:
                raise ValueError("Missing LR files for: {}".format(", ".join(missing[:5])))

        self.crop_size = crop_size
        self.scale = scale
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.hr_files)

    def _crop_pair(self, hr, lr):
        _, h, w = hr.shape
        if self.crop_size is None or h <= self.crop_size or w <= self.crop_size:
            return hr, lr
        if self.random_crop:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
        hr = hr[:, top:top + self.crop_size, left:left + self.crop_size]
        lr = lr[:, top:top + self.crop_size, left:left + self.crop_size]
        return hr, lr

    def __getitem__(self, index):
        hr = normalize_ct(_read_npz_slice(self.hr_files[index], self.random_crop), self.hu_min, self.hu_max)
        hr = torch.from_numpy(hr).unsqueeze(0).float()

        if self.lr_files is None:
            lr = F.interpolate(
                hr.unsqueeze(0),
                scale_factor=1.0 / self.scale,
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).squeeze(0)
            lr = torch.clamp(lr, 0.0, 1.0)
        else:
            lr = normalize_ct(_read_npz_slice(self.lr_files[index], self.random_crop), self.hu_min, self.hu_max)
            lr = torch.from_numpy(lr).unsqueeze(0).float()
            if lr.shape[-2:] != hr.shape[-2:]:
                lr = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode="bicubic", align_corners=False).squeeze(0)
                lr = torch.clamp(lr, 0.0, 1.0)

        hr, lr = self._crop_pair(hr, lr)

        if self.random_flip and random.random() < 0.5:
            hr = torch.flip(hr, dims=[2])
            lr = torch.flip(lr, dims=[2])
        if self.random_flip and random.random() < 0.5:
            hr = torch.flip(hr, dims=[1])
            lr = torch.flip(lr, dims=[1])
        return lr, hr

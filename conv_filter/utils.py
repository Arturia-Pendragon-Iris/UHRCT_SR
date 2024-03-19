import glob
import os
import numpy as np
from .filter import perform_filter
from torch.utils.data.dataset import Dataset
import torch
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandAffine)


train_transforms = Compose(
    [RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(np.pi / 8, np.pi / 8),
        scale_range=(-0.5, 0.5)),
     RandFlip(prob=0.5),
     RandRotate90(prob=0.5)
    ]
)


def refine_ct(ct_array):
    # k = np.random.randint(low=-1000, high=min(600, np.max(ct_array) - 400))
    # k = np.random.randint(low=0, high=240)
    ct_array = np.clip((ct_array + 1000) / 1600, 0, 1)

    return ct_array


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        total_list = []
        for i in os.listdir(dataset_dir):
            i_path = os.path.join(dataset_dir, i)
            sub_list = os.listdir(i_path)
            sub_list = [os.path.join(i_path, sub_list[x]) for x in range(len(sub_list))]
            total_list.extend(sub_list)

        self.file_list = total_list
        self.device = device

    def __getitem__(self, index):
        raw_array = np.load(self.file_list[index])["arr_0"]
        raw_array = refine_ct(raw_array[0])
        # print(raw_array.shape)
        raw_array = np.clip(train_transforms(raw_array[np.newaxis]), 0, 1)

        raw_tensor = torch.tensor(raw_array).clone().to(torch.float).to(self.device)
        filtered_tensor = torch.tensor(perform_filter(raw_array[0], enhance=False)[np.newaxis]).clone().\
            to(torch.float).to(self.device)
        return raw_tensor, filtered_tensor

    def __len__(self):
        return len(self.file_list)



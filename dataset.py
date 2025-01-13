import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torchvision.transforms as transforms

from typing import Any
from torch.utils.data import Dataset
from utils.util import pad_crop, read_file, find_arrays_with_object


class SatelliteDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        transform : transforms = None,
    ) -> None:
        if not os.path.exists(data_dir):
            raise ValueError(f'Provided data_dir: "{data_dir}" does not exist.')
        
        self.data_dir = data_dir
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_list, self.sampling_indices = pad_crop(read_file(self.mask_dir, True, 'mask_norm'), 224, True, None)
        self.image_list, _ = pad_crop(read_file(self.image_dir, True, 'linear_norm'), 224, False, self.sampling_indices)
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform = transform


        # Data Split
        num_samples = len(self.image_list)
        indices = list(range(num_samples))

        np.random.shuffle(indices)
        num_val_samples = int(self.val_ratio * num_samples)
        num_test_samples = int(self.test_ratio * num_samples)
        if self.split == "train":
            self.indices = indices[:-num_val_samples-num_test_samples]
        elif self.split == "val":
            self.indices = indices[-num_val_samples-num_test_samples:-num_test_samples]
        elif self.split == 'test':
            self.indices = indices[-num_test_samples:]
        else:
            raise ValueError("Invalid split value. Use 'train', 'val' or 'test'.")

 
    def __len__(self) -> int:
        return len(self.indices)


    def __getitem__(self, idx: Any) -> Any:
        img_idx = self.indices[idx]
        img = self.image_list[img_idx]
        mask = self.mask_list[img_idx]
        
        padded_img = np.pad(img, ((0,0),(16,16),(16,16)), 'constant', constant_values=0)
        padded_img = np.transpose(padded_img, (2,1,0))
        padded_mask = np.pad(mask, ((0,0),(16,16),(16,16)), 'constant', constant_values=0)
        padded_mask = np.transpose(padded_mask, (2,1,0))

        if self.transform:
            augmentations = self.transform(image=padded_img, mask=padded_mask)
            processed_img = augmentations["image"]
            processed_mask = augmentations["mask"]

        else:
            processed_img = transforms.ToTensor()(padded_img)
            processed_mask = transforms.ToTensor()(padded_mask)

        return processed_img, processed_mask


class InferenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        transform : transforms = None,
    ) -> None:
        if not os.path.exists(data_dir):
            raise ValueError(f'Provided data_dir: "{data_dir}" does not exist.')
        
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_list, _ = pad_crop(read_file(self.image_dir, True, 'linear_norm'), 224, False, None)
        self.mask_list, _ = pad_crop(read_file(self.mask_dir, True, 'mask_norm'), 224, False, None)
        self.transform = transform
        self.indices = list(range(len(self.image_list)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: Any) -> Any:
        img_idx = self.indices[idx]
        img = self.image_list[img_idx]
        mask = self.mask_list[img_idx]
        
        padded_img = np.pad(img, ((0,0),(16,16),(16,16)), 'constant', constant_values=0)
        padded_img = np.transpose(padded_img, (2,1,0))
        padded_mask = np.pad(mask, ((0,0),(16,16),(16,16)), 'constant', constant_values=0)
        padded_mask = np.transpose(padded_mask, (2,1,0))

        if self.transform:
            augmentations = self.transform(image=padded_img, mask=padded_mask)
            processed_img = augmentations["image"]
            processed_mask = augmentations["mask"]
        else:
            processed_img = transforms.ToTensor()(padded_img)
            processed_mask = transforms.ToTensor()(padded_mask)

        return processed_img, processed_mask
"""
dataset.py — Medical Image Segmentation Dataset
================================================
Compatible with:
  • Breast Ultrasound Images (BUSI) dataset
    https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
  • ISIC 2018 Skin Lesion Segmentation dataset
    https://challenge.isic-archive.com/data/#2018

Expected directory structure (BUSI example):
    data/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   └── ...
    ├── malignant/
    │   ├── malignant (1).png
    │   └── ...
    └── normal/
        └── ...

For ISIC, set isic_mode=True and point data_dir at the folder containing
ISIC_*_input.jpg and ISIC_*_segmentation.png pairs.
"""

import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


class BUSIDataset(Dataset):
    """
    Breast Ultrasound Images (BUSI) segmentation dataset.
    Loads image + corresponding _mask pairs from BUSI folder layout.
    """

    def __init__(self,
                 data_dir: str,
                 img_size: int = 256,
                 augment: bool = True,
                 categories: list = None):
        self.img_size = img_size
        self.augment  = augment
        self.samples  = []

        categories = categories or ["benign", "malignant"]
        for cat in categories:
            cat_dir = Path(data_dir) / cat
            if not cat_dir.exists():
                continue
            images = sorted(glob.glob(str(cat_dir / "*.png")))
            # Filter out mask files
            images = [f for f in images if "_mask" not in f]
            for img_path in images:
                mask_path = img_path.replace(".png", "_mask.png")
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        print(f"[BUSIDataset] Found {len(self.samples)} image-mask pairs "
              f"from categories: {categories}")

    def __len__(self):
        return len(self.samples)

    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def _augment(self, image: Image.Image, mask: Image.Image):
        """Joint spatial augmentations applied identically to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)
        # Random brightness / contrast (image only)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
        return image, mask

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("L")   # grayscale
        mask  = Image.open(mask_path).convert("L")

        image = self._resize(image)
        mask  = self._resize(mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        # Convert to tensors
        image_t = TF.to_tensor(image)                      # [1, H, W] in [0,1]
        mask_t  = TF.to_tensor(mask)                       # [1, H, W] in [0,1]
        mask_t  = (mask_t > 0.5).float()                   # binarise

        # Normalise image to zero-mean / unit-std (ImageNet-style for grayscale)
        image_t = TF.normalize(image_t, mean=[0.5], std=[0.5])

        return image_t, mask_t


class ISICDataset(Dataset):
    """
    ISIC 2018 Skin Lesion Segmentation dataset.
    Expects pairs: ISIC_XXXXXXX_input.jpg + ISIC_XXXXXXX_segmentation.png
    """

    def __init__(self,
                 data_dir: str,
                 img_size: int = 256,
                 augment: bool = True):
        self.img_size = img_size
        self.augment  = augment
        self.samples  = []

        images = sorted(glob.glob(str(Path(data_dir) / "*_input.jpg")))
        for img_path in images:
            mask_path = img_path.replace("_input.jpg", "_segmentation.png")
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        print(f"[ISICDataset] Found {len(self.samples)} image-mask pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("L")
        mask  = Image.open(mask_path).convert("L")

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize( (self.img_size, self.img_size), Image.NEAREST)

        image_t = TF.to_tensor(image)
        mask_t  = TF.to_tensor(mask)
        mask_t  = (mask_t > 0.5).float()

        image_t = TF.normalize(image_t, mean=[0.5], std=[0.5])
        return image_t, mask_t


def get_dataloaders(data_dir: str,
                    dataset_type: str = "busi",
                    img_size: int = 256,
                    batch_size: int = 8,
                    val_split: float = 0.2,
                    num_workers: int = 2):
    """
    Build train / validation DataLoaders with an 80/20 split.

    Args:
        data_dir      : Root directory of the dataset.
        dataset_type  : 'busi' or 'isic'.
        img_size      : Spatial resolution to resize images to (square).
        batch_size    : Mini-batch size.
        val_split     : Fraction of data to hold out for validation.
        num_workers   : DataLoader worker processes.

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import random_split, DataLoader

    if dataset_type.lower() == "busi":
        full_dataset = BUSIDataset(data_dir, img_size=img_size, augment=True)
    elif dataset_type.lower() == "isic":
        full_dataset = ISICDataset(data_dir, img_size=img_size, augment=True)
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Use 'busi' or 'isic'.")

    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Disable augmentation on the validation split
    if hasattr(val_set.dataset, "augment"):
        val_set.dataset.augment = False

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"[DataLoader] Train: {n_train} samples | Val: {n_val} samples")
    return train_loader, val_loader

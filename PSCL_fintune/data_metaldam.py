"""
MetalDAM dataset adapter for PSCL finetuning.

Bridges MetalDAM's patch format to PSCL's expected format:
  MetalDAM: 256x256 grayscale uint8 PNG, 5 classes (0-4) + 255 ignore
  PSCL:     376x376 RGB float32, 4 classes one-hot (4, H, W)

Class remapping (applied before resize):
  0 Background  -> 255 (ignore)
  1 Austenite   -> 0
  2 Matrix      -> 1
  3 MA          -> 2
  4 Precipitate -> 3
"""

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from data import get_one_hot

_TARGET = 376
_MEAN = (0.49139968, 0.48215841, 0.44653091)
_STD  = (0.24703223, 0.24348513, 0.26158784)

# Lookup table: index i maps to PSCL class _REMAP[i]
_REMAP = np.array([255, 0, 1, 2, 3], dtype=np.uint8)


def _load_image(path: str) -> np.ndarray:
    """Load grayscale PNG, resize to 376x376, replicate to (H, W, 3) uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (_TARGET, _TARGET), interpolation=cv2.INTER_LINEAR)
    return np.stack([img, img, img], axis=-1)   # (H, W, 3) uint8


def _load_mask(path: str) -> np.ndarray:
    """Load MetalDAM mask, remap classes, resize with INTER_NEAREST."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = _REMAP[mask]   # remap before resize — INTER_NEAREST on correct values
    mask = cv2.resize(mask, (_TARGET, _TARGET), interpolation=cv2.INTER_NEAREST)
    return mask   # uint8: values 0-3 + 255 (ignore)


def _mask_to_onehot(mask: np.ndarray) -> torch.Tensor:
    """Convert remapped mask to (4, H, W) float32 one-hot. Ignore pixels -> all zero."""
    ignore = (mask == 255)
    m = mask.copy()
    m[ignore] = 0                                    # temp: avoid out-of-range index
    gt = get_one_hot(m, num_classes=4)               # (H, W, 4) int64
    gt = gt.permute(2, 0, 1).float()                 # (4, H, W)
    gt[:, torch.from_numpy(ignore)] = 0.0            # zero all channels for ignored pixels
    return gt


def _supervised_transform(training: bool) -> transforms.Compose:
    jitter_d = 0.1
    if training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)],
                p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _moco_transform(jitter_d: float = 0.2) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply(
            [transforms.ColorJitter(
                0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)],
            p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ---------------------------------------------------------------------------
# 1. Supervised dataset — mirrors Data_preheat
# ---------------------------------------------------------------------------

class MetalDAMDataset(data.Dataset):
    """Supervised dataset for finetuning and evaluation.

    train -> returns (image [3,376,376], gt_onehot [4,376,376])
    val/test -> returns (image, gt_onehot, filename)
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        self.split = split
        self.img_dir  = os.path.join(data_dir, split, 'images_norm')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        self.filelist = sorted(f for f in os.listdir(self.img_dir) if f.endswith('.png'))
        self.transform = _supervised_transform(training=(split == 'train'))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fname = self.filelist[index]
        img_rgb = _load_image(os.path.join(self.img_dir, fname))
        mask    = _load_mask(os.path.join(self.mask_dir, fname))

        img = self.transform(img_rgb)
        gt  = _mask_to_onehot(mask)

        if self.split == 'train':
            rot_k = random.randint(1, 4)
            filp  = random.randint(1, 2)
            if filp == 1:
                img = torch.flip(img, dims=[1])
                gt  = torch.flip(gt,  dims=[1])
            img = torch.rot90(img, k=rot_k, dims=[1, 2])
            gt  = torch.rot90(gt,  k=rot_k, dims=[1, 2])
            return img, gt
        else:
            return img, gt, fname


# ---------------------------------------------------------------------------
# 2. Unlabeled MoCo dataset — mirrors MoCoData_preheat
# ---------------------------------------------------------------------------

class MetalDAMMoCoDataset(data.Dataset):
    """Unlabeled pool for PSCL self-supervised pretraining.

    Returns (img_cat [6,376,376], parent_id, rot_k, filp, r1, r2, r3, r4)
    where img_cat = cat([view_q, view_k], dim=0).
    """

    def __init__(self, data_dir: str, jitter_d: float = 0.2, random_c: float = 0.1):
        self.img_dir   = os.path.join(data_dir, 'train', 'images_norm')
        self.filelist  = sorted(f for f in os.listdir(self.img_dir) if f.endswith('.png'))
        self.random_c  = random_c
        self.transform = _moco_transform(jitter_d)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fname     = self.filelist[index]
        parent_id = fname.split('__')[0]    # e.g., "micrograph0"
        # model.py hardcodes np.eye(6): split('_')[0] must map to ≤5 unique values.
        # Bucket parent images into 4 groups so the prefix 'g0'..'g3' is always < 6.
        parent_num = int(''.join(filter(str.isdigit, parent_id)) or '0')
        id_str = f'g{parent_num % 4}_{parent_id}'
        img_rgb   = _load_image(os.path.join(self.img_dir, fname))

        img1 = self.transform(img_rgb)
        img2 = self.transform(img_rgb)

        rot_k = random.randint(1, 4)
        filp  = random.randint(1, 2)
        r1 = 1 - random.uniform(0, self.random_c)
        r2 = 1 - random.uniform(0, self.random_c)
        r3 = random.uniform(0, 1 - r1)
        r4 = random.uniform(0, 1 - r2)

        if filp == 1:
            img2 = torch.flip(img2, dims=[1])
        img2 = torch.rot90(img2, k=rot_k, dims=[1, 2])
        _, h, w = img2.shape
        img2 = img2[:, int(r3*h):int(r3*h+r1*h), int(r4*w):int(r4*w+r2*w)]
        img2 = transforms.Resize([h, w])(img2)

        img12 = torch.cat([img1, img2], dim=0)   # (6, H, W)
        return img12, id_str, rot_k, filp, r1, r2, r3, r4


# ---------------------------------------------------------------------------
# 3. Labeled guidance dataset — mirrors MoCoData_preheat_sup
# ---------------------------------------------------------------------------

class MetalDAMMoCoDatasetSup(data.Dataset):
    """Small labeled dataset used to guide patch sampling during PSCL pretraining.

    Returns (img_cat_gt [10,376,376], parent_id, rot_k, filp, r1, r2, r3, r4)
    where img_cat_gt = cat([view_q (3), view_k (3), gt_onehot*255 (4)], dim=0).

    Use the val split as guidance — it is labeled and from different parent images
    than the training unlabeled pool, preventing leakage.
    """

    def __init__(self, data_dir: str, sup_split: str = 'val',
                 jitter_d: float = 0.2, random_c: float = 0.1):
        self.img_dir   = os.path.join(data_dir, sup_split, 'images_norm')
        self.mask_dir  = os.path.join(data_dir, sup_split, 'masks')
        self.filelist  = sorted(f for f in os.listdir(self.img_dir) if f.endswith('.png'))
        self.random_c  = random_c
        self.transform = _moco_transform(jitter_d)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fname     = self.filelist[index]
        parent_id = fname.split('__')[0]
        parent_num = int(''.join(filter(str.isdigit, parent_id)) or '0')
        id_str = f'g{parent_num % 4}_{parent_id}'
        img_rgb   = _load_image(os.path.join(self.img_dir, fname))
        mask      = _load_mask(os.path.join(self.mask_dir, fname))

        img1 = self.transform(img_rgb)
        img2 = self.transform(img_rgb)

        # Build scaled one-hot (matches MoCoData_preheat_sup: values 0 or 255)
        ignore = (mask == 255)
        m = mask.copy()
        m[ignore] = 0
        gt = get_one_hot(m, num_classes=4) * 255          # (H, W, 4) scaled
        gt = gt.permute(2, 0, 1).float()                   # (4, H, W)
        gt[:, torch.from_numpy(ignore)] = 0.0

        rot_k = random.randint(1, 4)
        filp  = random.randint(1, 2)
        r1 = 1 - random.uniform(0, self.random_c)
        r2 = 1 - random.uniform(0, self.random_c)
        r3 = random.uniform(0, 1 - r1)
        r4 = random.uniform(0, 1 - r2)

        if filp == 1:
            img2 = torch.flip(img2, dims=[1])
            gt   = torch.flip(gt,   dims=[1])
        img2 = torch.rot90(img2, k=rot_k, dims=[1, 2])
        gt   = torch.rot90(gt,   k=rot_k, dims=[1, 2])
        _, h, w = img2.shape
        img2 = img2[:, int(r3*h):int(r3*h+r1*h), int(r4*w):int(r4*w+r2*w)]
        img2 = transforms.Resize([h, w])(img2)
        gt   = gt[:,  int(r3*h):int(r3*h+r1*h), int(r4*w):int(r4*w+r2*w)]
        gt   = transforms.Resize([h, w])(gt)

        img12G = torch.cat([img1, img2, gt], dim=0)   # (10, H, W)
        return img12G, id_str, rot_k, filp, r1, r2, r3, r4

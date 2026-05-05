"""
PyTorch Dataset for MetalDAM patch-based segmentation.

Returns (image [1,H,W] float32, mask [H,W] int64) pairs.
Images are expected to have been intensity-normalised by Node 07 (float32 PNGs
scaled to [0,1] stored as uint8; re-normalised here on load).
Masks are single-channel uint8 PNGs with class indices 0–4 and 255 (ignore).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IGNORE_LABEL: int = 255
N_CLASSES: int = 5


class MetalDAMDataset(Dataset):
    """Patch-level dataset for SEM micrograph segmentation.

    Parameters
    ----------
    metadata_csv:
        Path to ``metadata.csv`` produced by the preprocessing pipeline.
    split:
        One of ``'train'``, ``'val'``, ``'test'``.
    image_dir:
        Root directory under which patch images live.  Filenames are read
        from the ``filename`` column of the CSV.
    mask_dir:
        Root directory for mask patches (same filenames as images).
    transform:
        Albumentations ``Compose`` pipeline (from ``augmentations.py``).
        Must accept and return ``{"image": ..., "mask": ...}``.

    Examples
    --------
    >>> from src.datasets.augmentations import get_train_transforms
    >>> ds = MetalDAMDataset(
    ...     "metadata.csv", split="train",
    ...     image_dir="data/patches/train/images",
    ...     mask_dir="data/patches/train/masks",
    ...     transform=get_train_transforms(),
    ... )
    >>> img, mask = ds[0]
    >>> print(img.shape, mask.shape)   # (1, 256, 256)  (256, 256)
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform

        df = pd.read_csv(metadata_csv)
        # Keep only actual patch rows — they have patch_y/patch_x coordinates.
        # Parent-level metadata rows (from extract_scale, resolution_norm) have
        # no patch_y and must not be passed to the image loader.
        if "patch_y" in df.columns:
            df = df[df["patch_y"].notna()]
        self._df = df[df["split"] == split].reset_index(drop=True)
        if self._df.empty:
            raise ValueError(
                f"No rows found for split='{split}' in '{metadata_csv}'."
            )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        fname = row["filename"]

        # ── load image ────────────────────────────────────────────────────
        img_path = self.image_dir / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: '{img_path}'")
        # Node 07 saves normalised images as uint8; restore float32 [0,1].
        img_f32 = img.astype(np.float32) / 255.0

        # ── load mask ─────────────────────────────────────────────────────
        msk_path = self.mask_dir / fname
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: '{msk_path}'")

        # ── augmentation ──────────────────────────────────────────────────
        if self.transform is not None:
            # Albumentations expects uint8 image; rescale temporarily.
            img_u8 = (img_f32 * 255).clip(0, 255).astype(np.uint8)
            result = self.transform(image=img_u8, mask=mask)
            img_u8 = result["image"]
            mask   = result["mask"]
            img_f32 = img_u8.astype(np.float32) / 255.0

        # ── to tensors ────────────────────────────────────────────────────
        # Image: add channel dim → [1, H, W]
        img_tensor  = torch.from_numpy(img_f32).unsqueeze(0)
        # Mask: int64 for CrossEntropyLoss; ignore label stays 255
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return img_tensor, mask_tensor

    def get_class_weights(self) -> torch.Tensor:
        """Return inverse-frequency class weights as a float32 tensor.

        Weights are computed from the ``class_histogram`` column, excluding
        the ignore label (255).  Useful for initialising the ``weight``
        argument of ``nn.CrossEntropyLoss``.

        Returns
        -------
        torch.Tensor
            Shape ``(N_CLASSES,)``, dtype float32.
        """
        counts = np.zeros(N_CLASSES, dtype=np.float64)

        for hist_json in self._df["class_histogram"].dropna():
            try:
                hist: Dict[str, int] = json.loads(hist_json)
            except (json.JSONDecodeError, TypeError):
                continue
            for cls_str, cnt in hist.items():
                cls = int(cls_str)
                if cls < N_CLASSES:
                    counts[cls] += int(cnt)

        counts = np.maximum(counts, 1)   # avoid div-by-zero for absent classes
        weights = counts.sum() / (N_CLASSES * counts)
        return torch.tensor(weights, dtype=torch.float32)

"""
Training loop for U-Net segmentation on MetalDAM patches.

Loss: Dice + weighted CrossEntropyLoss (ignore_index=255).
Metrics tracked per epoch: loss, mean IoU (macro, ignoring class 255).

Usage
-----
    python src/train.py --config configs/preprocess.yaml
    python src/train.py --config configs/preprocess.yaml --epochs 50 --lr 1e-4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work when
# this script is invoked directly (python src/train.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.augmentations import get_train_transforms, get_val_transforms
from src.datasets.metaldam_dataset import IGNORE_LABEL, N_CLASSES, MetalDAMDataset
from src.models.unet import UNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── loss functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, ignoring label 255."""

    def __init__(self, n_classes: int = N_CLASSES, smooth: float = 1.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)   targets: (B, H, W) int64
        probs = F.softmax(logits, dim=1)
        valid = targets != IGNORE_LABEL                        # (B, H, W) bool

        dice_per_class = torch.zeros(self.n_classes, device=logits.device)
        for c in range(self.n_classes):
            p = probs[:, c][valid]
            t = (targets[valid] == c).float()
            intersection = (p * t).sum()
            dice_per_class[c] = (2.0 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth
            )

        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """0.5 × Dice + 0.5 × weighted CrossEntropy."""

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights,
                                        ignore_index=IGNORE_LABEL)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# ── metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def mean_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = N_CLASSES,
) -> float:
    """Macro-mean IoU across valid classes, ignoring label 255."""
    preds = logits.argmax(dim=1)                                # (B, H, W)
    valid = targets != IGNORE_LABEL

    ious = []
    for c in range(n_classes):
        pred_c = (preds == c) & valid
        true_c = (targets == c) & valid
        intersection = (pred_c & true_c).sum().item()
        union        = (pred_c | true_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


# ── one epoch ────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> Tuple[float, float]:
    model.train(training)
    total_loss = 0.0
    total_iou  = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            loss   = criterion(logits, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou  += mean_iou(logits, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


# ── training entry point ──────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    patch_dir   = Path(cfg.get("patch_dir", "data/patches"))
    meta_csv    = cfg.get("metadata_csv", "metadata.csv")
    out_dir     = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = MetalDAMDataset(
        meta_csv, split="train",
        image_dir=patch_dir / "train" / "images",
        mask_dir= patch_dir / "train" / "masks",
        transform=get_train_transforms(),
    )
    val_ds = MetalDAMDataset(
        meta_csv, split="val",
        image_dir=patch_dir / "val" / "images",
        mask_dir= patch_dir / "val" / "masks",
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = UNet(in_channels=1, n_classes=N_CLASSES).to(device)
    logger.info(
        "Parameters: %s M",
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}",
    )

    # ── loss & optimiser ──────────────────────────────────────────────────
    class_weights = train_ds.get_class_weights().to(device)
    criterion = CombinedLoss(class_weights)

    lr = float(cfg.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("epochs", 50),
        eta_min=lr / 100,
    )

    # ── training loop ─────────────────────────────────────────────────────
    best_val_iou = 0.0
    epochs = int(cfg.get("epochs", 50))

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_iou = run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  |  "
            "train loss %.4f  iou %.4f  |  "
            "val loss %.4f  iou %.4f  |  %.1fs",
            epoch, epochs,
            train_loss, train_iou,
            val_loss, val_iou,
            elapsed,
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt = out_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_loss": val_loss,
                    "cfg": cfg,
                },
                ckpt,
            )
            logger.info("  ✓ Saved checkpoint (val_iou=%.4f) → %s", val_iou, ckpt)

    logger.info("Training complete.  Best val IoU: %.4f", best_val_iou)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Train U-Net on MetalDAM patches.")
    parser.add_argument("--config",   default="configs/preprocess.yaml")
    parser.add_argument("--epochs",   type=int,   default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    train(cfg)

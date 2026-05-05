"""
Node 06 — Overlapping patch tiling for image/mask pairs.

Slides a fixed window over each image with configurable stride and drops
patches that contain insufficient foreground content.  Saves images and
masks into mirrored output subdirectories and returns structured metadata
ready to be appended to metadata.csv.

Split rule: patches inherit their parent image's split assignment.
Never split at patch level — see parent-level-split rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_patches(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    min_foreground: float = 0.05,
    parent_file: str = "",
    um_per_pixel: float = 0.0,
) -> List[Dict[str, Any]]:
    """Slide a window over an image/mask pair and return valid patches.

    A patch is kept when the fraction of pixels with class index != 0
    (i.e. non-background) is >= *min_foreground*.

    Parameters
    ----------
    img:
        H×W or H×W×C uint8 image.
    mask:
        H×W uint8 class-index mask.  Must share spatial dims with *img*.
    patch_size:
        Side length of the square patch window in pixels.
    stride:
        Step between consecutive patch origins.  ``stride < patch_size``
        produces overlapping patches.
    min_foreground:
        Minimum fraction of non-background pixels required to keep a patch.
        Background is defined as class index 0.
    parent_file:
        Stem of the source image.  Recorded in each patch's metadata so
        patches can be traced back to their parent for split assignment.
    um_per_pixel:
        Physical resolution of the *source* image in µm/pixel.  Recorded
        unchanged in every child patch (resolution normalisation is done in
        Node 05 before this step).

    Returns
    -------
    list of dict
        Each dict contains:

        * ``img_patch``    — np.ndarray, patch_size×patch_size×C uint8
        * ``mask_patch``   — np.ndarray, patch_size×patch_size uint8
        * ``parent_file``  — str, source image stem
        * ``y``            — int, top-left row in parent image
        * ``x``            — int, top-left col in parent image
        * ``um_per_pixel`` — float
        * ``class_histogram`` — dict, {str(class_idx): pixel_count}

    Raises
    ------
    ValueError
        If img/mask spatial dims differ or patch_size > image dimensions.

    Examples
    --------
    >>> patches = extract_patches(img, mask, patch_size=256, stride=128,
    ...                           parent_file="sample_01", um_per_pixel=0.012)
    >>> print(len(patches), "patches extracted")
    """
    _validate(img, mask, patch_size)

    h, w = img.shape[:2]
    patches: List[Dict[str, Any]] = []

    total_windows = 0
    skipped = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            total_windows += 1

            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            fg_fraction = _foreground_fraction(mask_patch)
            if fg_fraction < min_foreground:
                skipped += 1
                continue

            img_patch = img[y : y + patch_size, x : x + patch_size]

            patches.append(
                {
                    "img_patch": img_patch.copy(),
                    "mask_patch": mask_patch.copy(),
                    "parent_file": parent_file,
                    "y": y,
                    "x": x,
                    "um_per_pixel": um_per_pixel,
                    "class_histogram": _class_histogram(mask_patch),
                }
            )

    logger.info(
        "[patch_extract] '%s' — %d/%d windows kept "
        "(skipped %d with fg < %.1f%%)",
        parent_file or "?",
        len(patches),
        total_windows,
        skipped,
        min_foreground * 100,
    )

    return patches


def save_patches(
    patches: List[Dict[str, Any]],
    out_dir: str | Path,
) -> List[Dict[str, Any]]:
    """Save patch images and masks to mirrored subdirectory trees.

    Directory layout created inside *out_dir*::

        out_dir/
          images/   ← BGR images saved as PNG
          masks/    ← single-channel uint8 class-index masks saved as PNG

    Filenames encode the parent stem, top-left coordinates, and patch size
    for unambiguous provenance: ``{parent}__y{y:05d}_x{x:05d}.png``.

    Parameters
    ----------
    patches:
        List returned by :func:`extract_patches`.
    out_dir:
        Root output directory.  ``images/`` and ``masks/`` sub-dirs are
        created automatically.

    Returns
    -------
    list of dict
        Flat metadata records (no array data) suitable for appending to
        ``metadata.csv``.  Contains all patch fields except the raw arrays.

    Examples
    --------
    >>> records = save_patches(patches, "data/patches/train")
    >>> import pandas as pd
    >>> pd.DataFrame(records).to_csv("metadata.csv", index=False)
    """
    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for p in patches:
        fname = f"{p['parent_file']}__y{p['y']:05d}_x{p['x']:05d}.png"

        cv2.imwrite(str(img_out / fname), p["img_patch"])

        # Single-channel PNG — imwrite writes uint8 as-is when ndim == 2.
        cv2.imwrite(str(msk_out / fname), p["mask_patch"])

        h, w = p["img_patch"].shape[:2]
        records.append(
            {
                "filename": fname,
                "parent_id": p["parent_file"],
                "um_per_pixel": p["um_per_pixel"],
                "height": h,
                "width": w,
                "patch_y": p["y"],
                "patch_x": p["x"],
                "class_histogram": json.dumps(p["class_histogram"]),
            }
        )

    logger.info(
        "[patch_extract] Saved %d patches to '%s'.",
        len(records),
        out_dir,
    )
    return records


# ── private helpers ──────────────────────────────────────────────────────────

def _validate(img: np.ndarray, mask: np.ndarray, patch_size: int) -> None:
    ih, iw = img.shape[:2]
    mh, mw = mask.shape[:2]
    if (ih, iw) != (mh, mw):
        raise ValueError(
            f"img ({ih}×{iw}) and mask ({mh}×{mw}) spatial dims must match."
        )
    if patch_size > ih or patch_size > iw:
        raise ValueError(
            f"patch_size={patch_size} exceeds image dimensions ({ih}×{iw})."
        )


def _foreground_fraction(mask_patch: np.ndarray) -> float:
    """Return fraction of pixels with class index != 0."""
    return float((mask_patch != 0).sum()) / mask_patch.size


def _class_histogram(mask_patch: np.ndarray) -> Dict[str, int]:
    """Return {str(class_index): pixel_count} for non-zero classes."""
    values, counts = np.unique(mask_patch, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 06: extract overlapping patches from all image/mask pairs."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--split", default=None,
                        help="Process only this split (train/val/test).")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    meta_path = Path(cfg.get("metadata_csv", "metadata.csv"))
    img_dir   = Path(cfg.get("normed_image_dir",  "data/processed/images"))
    msk_dir   = Path(cfg.get("normed_mask_dir",   "data/processed/masks"))
    patch_dir = Path(cfg.get("patch_dir",         "data/patches"))
    patch_size    = int(cfg.get("patch_size",    256))
    stride        = int(cfg.get("stride",        128))
    min_fg        = float(cfg.get("min_foreground", 0.05))

    df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    # Build a quick lookup: parent_id → {split, um_per_pixel}
    parent_meta: dict[str, dict] = {}
    if not df.empty and "parent_id" in df.columns:
        for _, row in df.drop_duplicates("parent_id").iterrows():
            parent_meta[row["parent_id"]] = {
                "split": row.get("split", "train"),
                "um_per_pixel": row.get("um_per_pixel", 0.0),
            }

    all_records: list[dict] = []

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in img_paths:
        msk_path = msk_dir / (img_path.stem + ".png")
        if not msk_path.exists():
            logger.warning("[patch_extract] No mask for '%s'; skipping.", img_path.name)
            continue

        img  = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            logger.error("[patch_extract] Cannot read '%s'; skipping.", img_path.name)
            continue

        stem = img_path.stem
        meta  = parent_meta.get(stem, {"split": "train", "um_per_pixel": 0.0})
        split = meta["split"]
        um    = meta["um_per_pixel"]

        if args.split and split != args.split:
            continue

        patches = extract_patches(
            img, mask,
            patch_size=patch_size,
            stride=stride,
            min_foreground=min_fg,
            parent_file=stem,
            um_per_pixel=um,
        )

        out = patch_dir / split
        records = save_patches(patches, out)

        for rec in records:
            rec["split"] = split

        all_records.extend(records)

    if all_records:
        new_df = pd.DataFrame(all_records)
        if not df.empty:
            # Append without duplicating existing patch rows.
            existing_fnames = set(df["filename"]) if "filename" in df.columns else set()
            new_df = new_df[~new_df["filename"].isin(existing_fnames)]
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_csv(meta_path, index=False)
        logger.info("[patch_extract] Appended %d records to %s.", len(all_records), meta_path)

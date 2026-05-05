"""
Node 05 — Rescale images and masks to a common µm/pixel resolution.

All images in a dataset may originate from different SEM magnifications.
This node resamples every pair to a shared target_um_per_px so that
downstream patch extraction and training operate on a consistent physical scale.

Mask-interpolation rule: masks are ALWAYS resized with cv2.INTER_NEAREST.
Never use INTER_LINEAR, INTER_CUBIC, or INTER_AREA on class-index masks —
interpolating class labels creates phantom classes along boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# When |scale_factor - 1| < this, skip the resize entirely.
_SCALE_EPSILON = 1e-4


def normalize_resolution(
    img: np.ndarray,
    mask: np.ndarray,
    src_um_per_px: float,
    tgt_um_per_px: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample an image/mask pair to a target physical resolution.

    Parameters
    ----------
    img:
        H×W or H×W×C uint8 image (SEM micrograph, any channel count).
    mask:
        H×W uint8 class-index mask.  Must match ``img`` spatial dimensions.
    src_um_per_px:
        Current resolution in µm per pixel (> 0).
    tgt_um_per_px:
        Target resolution in µm per pixel (> 0).  Use ``null`` / ``None``
        to derive the dataset median externally and pass it here.

    Returns
    -------
    resized_img : np.ndarray
        Resampled image at target resolution.
    resized_mask : np.ndarray
        Resampled mask at target resolution (INTER_NEAREST, uint8).
    new_um_per_px : float
        Effective µm/pixel of the output (== *tgt_um_per_px* unless skipped).

    Raises
    ------
    ValueError
        If resolutions are non-positive, or if img/mask spatial dims differ.

    Examples
    --------
    >>> img_out, mask_out, um = normalize_resolution(img, mask, 0.012, 0.010)
    >>> print(img_out.shape, mask_out.shape, um)
    """
    _validate(img, mask, src_um_per_px, tgt_um_per_px)

    scale = src_um_per_px / tgt_um_per_px

    if abs(scale - 1.0) < _SCALE_EPSILON:
        logger.debug(
            "[resolution_norm] scale=%.6f ≈ 1; skipping resize.", scale
        )
        return img.copy(), mask.copy(), tgt_um_per_px

    h, w = img.shape[:2]
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    new_size = (new_w, new_h)  # cv2 takes (width, height)

    # Downscale: INTER_AREA averages pixels → suppresses aliasing.
    # Upscale:   INTER_LINEAR is smooth and fast.
    img_interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_img = cv2.resize(img, new_size, interpolation=img_interp)

    # Masks must always use INTER_NEAREST — see mask-interpolation rule.
    resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

    logger.info(
        "[resolution_norm] %.6f → %.6f µm/px  |  "
        "(%d×%d) → (%d×%d)  scale=%.4f  img_interp=%s",
        src_um_per_px, tgt_um_per_px,
        h, w, new_h, new_w,
        scale,
        "INTER_AREA" if img_interp == cv2.INTER_AREA else "INTER_LINEAR",
    )

    return resized_img, resized_mask, tgt_um_per_px


# ── batch processing ─────────────────────────────────────────────────────────

def batch_normalize(
    metadata_csv: str | Path,
    processed_img_dir: str | Path,
    mask_dir: str | Path,
    out_img_dir: str | Path,
    out_mask_dir: str | Path,
    tgt_um_per_px: float | None = None,
) -> pd.DataFrame:
    """Normalize all image/mask pairs listed in *metadata_csv*.

    Reads ``filename`` and ``um_per_pixel`` from the CSV.  When
    *tgt_um_per_px* is ``None``, the dataset median ``um_per_pixel`` is used
    as the target (recommended: lets the dataset self-normalise).

    Updates ``um_per_pixel`` in the CSV to *tgt_um_per_px* for every
    successfully processed row.  Never deletes other columns.

    Parameters
    ----------
    metadata_csv:
        Path to ``metadata.csv``.
    processed_img_dir:
        Directory containing cropped SEM images (Node 02 output).
    mask_dir:
        Directory containing uint8 class-index masks (Node 03 output).
    out_img_dir:
        Destination for resolution-normalised images.
    out_mask_dir:
        Destination for resolution-normalised masks.
    tgt_um_per_px:
        Target µm/pixel.  ``None`` → use dataset median.

    Returns
    -------
    pd.DataFrame
        Updated metadata with new ``um_per_pixel`` values.

    Raises
    ------
    FileNotFoundError
        If *metadata_csv* does not exist.
    ValueError
        If required CSV columns are missing.
    """
    meta_path = Path(metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: '{meta_path}'")

    df = pd.read_csv(meta_path)
    _require_columns(df, ["filename", "um_per_pixel"])

    Path(out_img_dir).mkdir(parents=True, exist_ok=True)
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)

    if tgt_um_per_px is None:
        tgt_um_per_px = float(df["um_per_pixel"].median())
        logger.info(
            "[resolution_norm] tgt_um_per_px not set — using dataset median: %.6f",
            tgt_um_per_px,
        )

    errors: list[str] = []
    updated_rows: list[int] = []

    for i, row in df.iterrows():
        stem = Path(str(row["filename"])).stem
        src_um = float(row["um_per_pixel"])

        img_path = _find_file(Path(processed_img_dir), stem, {".jpg", ".jpeg", ".png"})
        msk_path = _find_file(Path(mask_dir), stem, {".png"})

        if img_path is None:
            errors.append(f"  [missing-image]  '{stem}' not found in {processed_img_dir}")
            continue
        if msk_path is None:
            errors.append(f"  [missing-mask]   '{stem}' not found in {mask_dir}")
            continue
        if pd.isna(src_um) or src_um <= 0:
            errors.append(f"  [bad-um]         '{stem}' has invalid um_per_pixel={src_um!r}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            errors.append(f"  [unreadable]     '{img_path}'")
            continue
        if mask is None:
            errors.append(f"  [unreadable]     '{msk_path}'")
            continue

        try:
            r_img, r_mask, new_um = normalize_resolution(
                img, mask, src_um, tgt_um_per_px
            )
        except ValueError as exc:
            errors.append(f"  [resize-error]   '{stem}': {exc}")
            continue

        cv2.imwrite(str(Path(out_img_dir) / img_path.name), r_img)
        # Masks are saved as-is (uint8 class indices) — no colour conversion.
        cv2.imwrite(str(Path(out_mask_dir) / msk_path.name), r_mask)

        df.at[i, "um_per_pixel"] = new_um
        df.at[i, "height"] = r_img.shape[0]
        df.at[i, "width"] = r_img.shape[1]
        updated_rows.append(i)

    df.to_csv(meta_path, index=False)
    logger.info(
        "[resolution_norm] Updated %d/%d rows in %s (target=%.6f µm/px).",
        len(updated_rows), len(df), meta_path, tgt_um_per_px,
    )

    if errors:
        error_block = "\n".join(errors)
        raise RuntimeError(
            f"batch_normalize completed with {len(errors)} error(s):\n{error_block}"
        )

    return df


# ── private helpers ──────────────────────────────────────────────────────────

def _validate(
    img: np.ndarray,
    mask: np.ndarray,
    src_um: float,
    tgt_um: float,
) -> None:
    if src_um <= 0 or tgt_um <= 0:
        raise ValueError(
            f"um_per_px values must be positive; got src={src_um}, tgt={tgt_um}."
        )
    if img.ndim < 2:
        raise ValueError(f"img must be at least 2-D, got shape {img.shape}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (H×W), got shape {mask.shape}.")
    ih, iw = img.shape[:2]
    mh, mw = mask.shape[:2]
    if (ih, iw) != (mh, mw):
        raise ValueError(
            f"img ({ih}×{iw}) and mask ({mh}×{mw}) spatial dims must match."
        )


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {missing}")


def _find_file(directory: Path, stem: str, exts: set[str]) -> Path | None:
    for ext in exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 05: normalise all pairs to a common µm/pixel resolution."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    tgt = cfg.get("target_um_per_px") or None  # null in YAML → None

    batch_normalize(
        metadata_csv=cfg.get("metadata_csv", "metadata.csv"),
        processed_img_dir=cfg.get("processed_image_dir", "data/processed/images"),
        mask_dir=cfg.get("processed_mask_dir", "data/processed/masks"),
        out_img_dir=cfg.get("normed_image_dir", "data/processed/images"),
        out_mask_dir=cfg.get("normed_mask_dir", "data/processed/masks"),
        tgt_um_per_px=tgt,
    )

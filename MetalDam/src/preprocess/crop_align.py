"""
Node 02 — Info-bar crop and image/label pair alignment.

Crops the SEM micrograph to exactly label.shape[0] rows.
Labels are already bar-free; their height is the authoritative crop target.

Pipeline rule: always run AFTER extract_scale.py (Node 01).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum row-variance considered "micrograph content" (not flat info bar).
_VAR_THRESHOLD = 50.0
# Number of consecutive low-variance rows that confirm the bar boundary.
_CONSECUTIVE_LOW = 3


def crop_to_label_height(
    img_path: str | Path,
    label_path: str | Path,
    out_dir: str | Path,
) -> int:
    """Crop a SEM micrograph to exactly the height of its label image.

    The SEM info bar occupies the bottom rows of the raw micrograph but is
    absent from the coloured label.  This function:

    1. Loads the micrograph (BGR) and its RGB label.
    2. Auto-detects the info bar's top edge by scanning rows from the bottom
       and finding the first row whose variance drops below ``_VAR_THRESHOLD``
       for ``_CONSECUTIVE_LOW`` consecutive rows.  This is used only for
       logging; the actual crop height is always ``label.shape[0]``.
    3. Crops the micrograph to ``label.shape[0]`` rows from the top.
    4. Asserts the cropped image shape equals the label shape; raises
       ``ValueError`` with full details if not.
    5. Saves the cropped image to *out_dir* (same filename as input) and
       returns the crop row index used.

    Parameters
    ----------
    img_path:
        Path to the original SEM micrograph (.jpg).  Info bar must still be
        present — run extract_scale.py first.
    label_path:
        Path to the corresponding coloured label (.png), already bar-free.
    out_dir:
        Directory where the cropped image is saved.  Created if absent.

    Returns
    -------
    int
        The row index at which the image was cropped (== ``label.shape[0]``).

    Raises
    ------
    FileNotFoundError
        If either *img_path* or *label_path* cannot be loaded.
    ValueError
        If the cropped image shape does not match the label shape.

    Examples
    --------
    >>> crop_row = crop_to_label_height(
    ...     "data/raw/images/sample_01.jpg",
    ...     "data/raw/coloured_labels/sample_01.png",
    ...     "data/processed/images/",
    ... )
    >>> print(f"Cropped at row {crop_row}")
    """
    img_path = Path(img_path)
    label_path = Path(label_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = _load_bgr(img_path)
    label = _load_rgb(label_path)

    detected_edge = _detect_bar_top_edge(img)
    logger.debug(
        "[crop_align] %s — detected info-bar edge at row %d",
        img_path.name, detected_edge,
    )

    # label.shape[0] is the authoritative crop height (see crop-height rule).
    crop_row = label.shape[0]
    img_cropped = img[:crop_row, :]

    _assert_shapes_match(img_cropped, label, img_path, label_path)

    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img_cropped)
    logger.info(
        "[crop_align] %s — cropped at row %d (detected edge: %d) → %s",
        img_path.name, crop_row, detected_edge, out_path,
    )

    return crop_row


# ── private helpers ──────────────────────────────────────────────────────────

def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: '{path}'")
    return img


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read label: '{path}'")
    # Keep as BGR internally; shape is all we need here.
    return img


def _detect_bar_top_edge(img: np.ndarray) -> int:
    """Return the row index of the info bar's top edge.

    Scans upward from the bottom row.  The info bar has near-uniform rows
    (low variance); the micrograph content above it has high variance.
    Returns the first row from the bottom that starts a run of
    ``_CONSECUTIVE_LOW`` low-variance rows.

    Falls back to ``img.shape[0]`` (no bar detected) so the caller can
    still proceed using ``label.shape[0]`` as the crop target.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = gray.shape[0]

    # Per-row variance: shape (H,)
    row_var = gray.var(axis=1)

    consecutive = 0
    for row in range(h - 1, -1, -1):
        if row_var[row] < _VAR_THRESHOLD:
            consecutive += 1
            if consecutive >= _CONSECUTIVE_LOW:
                return row  # top of the low-variance run
        else:
            consecutive = 0

    logger.warning(
        "[crop_align] Info-bar edge not detected; image may already be cropped."
    )
    return h  # sentinel: no bar found


def _assert_shapes_match(
    img_cropped: np.ndarray,
    label: np.ndarray,
    img_path: Path,
    label_path: Path,
) -> None:
    """Raise ValueError if cropped image H×W does not match label H×W."""
    ih, iw = img_cropped.shape[:2]
    lh, lw = label.shape[:2]
    if (ih, iw) != (lh, lw):
        raise ValueError(
            f"Shape mismatch after crop:\n"
            f"  image '{img_path.name}' cropped to ({ih}, {iw})\n"
            f"  label '{label_path.name}' is       ({lh}, {lw})\n"
            "Verify that the label is bar-free and was derived from this image."
        )


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 02: crop SEM images to label height."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    raw_img_dir = Path(cfg.get("raw_image_dir", "data/raw/images"))
    raw_lbl_dir = Path(cfg.get("raw_label_dir", "data/raw/coloured_labels"))
    out_image_dir = Path(cfg.get("processed_image_dir", "data/processed/images"))

    img_paths = sorted(raw_img_dir.glob("*.jpg")) + sorted(raw_img_dir.glob("*.png"))
    for img_p in img_paths:
        lbl_p = raw_lbl_dir / (img_p.stem + ".png")
        if not lbl_p.exists():
            logger.warning("[crop_align] No label for '%s'; skipping.", img_p.name)
            continue
        try:
            row = crop_to_label_height(img_p, lbl_p, out_image_dir)
            logger.info("[crop_align] ✓ %s → row %d", img_p.name, row)
        except (ValueError, FileNotFoundError) as exc:
            logger.error("[crop_align] ✗ %s: %s", img_p.name, exc)

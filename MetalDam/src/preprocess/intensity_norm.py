"""
Node 07 — Per-image and dataset-wide intensity normalisation.

SEM micrographs are effectively grayscale.  All methods operate on a
single-channel 2-D array and return float32 in [0, 1].

Three methods are supported:
  zscore  — per-image zero-mean / unit-std, clipped to [-3, 3], scaled to [0, 1].
  clahe   — contrast-limited adaptive histogram equalisation, then [0, 1] float.
  minmax  — simple min-max stretch to [0, 1].

For dataset-wide z-score (training), use compute_dataset_stats() to obtain
a global (mean, std) and pass them directly instead of per-image statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_METHODS = ("zscore", "clahe", "minmax")
_ZSCORE_CLIP = 3.0          # sigma clip range
_CLAHE_CLIP  = 2.0          # clip limit for CLAHE
_CLAHE_GRID  = (8, 8)       # tile grid size for CLAHE


def normalize_intensity(
    img: np.ndarray,
    method: str = "zscore",
    *,
    global_mean: float | None = None,
    global_std: float | None = None,
) -> np.ndarray:
    """Normalise a single-channel SEM image to float32 in [0, 1].

    Parameters
    ----------
    img:
        H×W uint8 or float array (grayscale).  If 3-channel BGR is passed,
        it is converted to grayscale automatically.
    method:
        One of ``'zscore'``, ``'clahe'``, or ``'minmax'``.
    global_mean:
        When provided together with *global_std*, used instead of the
        per-image mean for z-score normalisation.  Ignored by other methods.
    global_std:
        Paired with *global_mean*.  Must be > 0.

    Returns
    -------
    np.ndarray
        H×W float32 array with values in [0, 1].

    Raises
    ------
    ValueError
        If *method* is not one of the supported options, or if the image
        cannot be converted to grayscale.

    Examples
    --------
    >>> from src.preprocess.intensity_norm import normalize_intensity
    >>> import cv2
    >>> img = cv2.imread("data/patches/train/images/sample_01__y00000_x00000.png",
    ...                  cv2.IMREAD_GRAYSCALE)
    >>> out = normalize_intensity(img, method="zscore")
    >>> print(out.dtype, out.min(), out.max())
    float32 0.0 1.0
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {_METHODS}."
        )

    gray = _to_gray_float32(img)

    if method == "zscore":
        return _zscore(gray, global_mean, global_std)
    if method == "clahe":
        return _clahe(img)
    if method == "minmax":
        return _minmax(gray)


def compute_dataset_stats(
    img_paths: Iterable[str | Path],
) -> Tuple[float, float]:
    """Compute the global mean and std across all images in the dataset.

    Uses Welford's online algorithm to accumulate statistics in a single
    pass without loading all images into memory simultaneously.

    Parameters
    ----------
    img_paths:
        Iterable of paths to grayscale or BGR images.

    Returns
    -------
    (mean, std) : tuple of float
        Values are in the original uint8 scale [0, 255].  Divide your images
        by 255 first if you want statistics in [0, 1] space, or normalise
        using these values directly before the [0, 255] → float conversion.

    Raises
    ------
    RuntimeError
        If no images can be read.

    Examples
    --------
    >>> from pathlib import Path
    >>> paths = sorted(Path("data/patches/train/images").glob("*.png"))
    >>> mean, std = compute_dataset_stats(paths)
    >>> print(f"Dataset mean={mean:.2f}  std={std:.2f}")
    """
    # Welford accumulator — numerically stable, single-pass.
    n_pixels: int = 0
    welford_mean: float = 0.0
    welford_m2: float = 0.0

    n_images = 0
    for path in img_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logger.warning("[intensity_norm] Cannot read '%s'; skipping.", path)
            continue

        pixels = gray.astype(np.float64).ravel()
        for px in _chunked(pixels, chunk=65536):
            for val in px:
                n_pixels += 1
                delta = val - welford_mean
                welford_mean += delta / n_pixels
                welford_m2 += delta * (val - welford_mean)

        n_images += 1
        if n_images % 50 == 0:
            logger.info(
                "[intensity_norm] Stats pass: %d images, running mean=%.2f",
                n_images, welford_mean,
            )

    if n_pixels == 0:
        raise RuntimeError(
            "compute_dataset_stats: no images could be read. "
            "Check that img_paths is non-empty and paths are valid."
        )

    std = float(np.sqrt(welford_m2 / n_pixels))
    mean = float(welford_mean)

    logger.info(
        "[intensity_norm] Dataset stats (%d images, %d px): mean=%.4f  std=%.4f",
        n_images, n_pixels, mean, std,
    )
    return mean, std


# ── normalisation kernels ────────────────────────────────────────────────────

def _zscore(
    gray: np.ndarray,
    global_mean: float | None,
    global_std: float | None,
) -> np.ndarray:
    """Z-score normalise, clip ±3σ, rescale to [0, 1]."""
    use_global = global_mean is not None and global_std is not None
    mean = float(global_mean) if use_global else float(gray.mean())
    std  = float(global_std)  if use_global else float(gray.std())

    if std < 1e-6:
        logger.warning(
            "[intensity_norm] Near-zero std (%.2e); returning zeros.", std
        )
        return np.zeros_like(gray, dtype=np.float32)

    norm = (gray - mean) / std
    norm = np.clip(norm, -_ZSCORE_CLIP, _ZSCORE_CLIP)
    # Rescale [-3, 3] → [0, 1].
    return ((norm + _ZSCORE_CLIP) / (2.0 * _ZSCORE_CLIP)).astype(np.float32)


def _clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE and return float32 [0, 1]."""
    # Accept BGR or grayscale input.
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # CLAHE operates on uint8.
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    equalized = clahe.apply(gray)
    return (equalized.astype(np.float32) / 255.0)


def _minmax(gray: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 1]."""
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        logger.warning(
            "[intensity_norm] Constant image (min==max=%.2f); returning zeros.", lo
        )
        return np.zeros_like(gray, dtype=np.float32)
    return ((gray - lo) / (hi - lo)).astype(np.float32)


# ── private utilities ────────────────────────────────────────────────────────

def _to_gray_float32(img: np.ndarray) -> np.ndarray:
    """Convert any supported input to float32 grayscale."""
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ValueError(
                f"Expected 1- or 3-channel image, got {img.shape[2]} channels."
            )
    if img.ndim != 2:
        raise ValueError(f"Cannot convert image with shape {img.shape} to grayscale.")
    return img.astype(np.float32)


def _chunked(arr: np.ndarray, chunk: int):
    """Yield successive *chunk*-size views of a 1-D array."""
    for start in range(0, len(arr), chunk):
        yield arr[start : start + chunk]


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 07: normalise patch intensities."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only compute and print dataset statistics, do not write files.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    method      = cfg.get("normalize_method", "zscore")
    patch_dir   = Path(cfg.get("patch_dir", "data/patches"))
    train_imgs  = sorted((patch_dir / "train" / "images").glob("*.png"))

    if args.stats_only or method == "zscore":
        mean, std = compute_dataset_stats(train_imgs)
        print(f"Global mean : {mean:.4f}")
        print(f"Global std  : {std:.4f}")
        if args.stats_only:
            raise SystemExit(0)
    else:
        mean = std = None

    for split in ("train", "val", "test"):
        in_dir  = patch_dir / split / "images"
        out_dir = patch_dir / split / "images_norm"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(in_dir.glob("*.png"))
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("[intensity_norm] Cannot read '%s'", p)
                continue
            normed = normalize_intensity(
                img, method=method,
                global_mean=mean, global_std=std,
            )
            out_path = out_dir / p.name
            cv2.imwrite(str(out_path), (normed * 255).astype(np.uint8))

        logger.info("[intensity_norm] %s — normalised %d images → %s", split, len(paths), out_dir)

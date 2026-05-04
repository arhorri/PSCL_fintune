"""
Node 03 — RGB label → single-channel class-index mask.

Converts coloured_labels PNG files (H×W×3 RGB) into H×W uint8 class maps
where each pixel holds its class index, or 255 for unmatched pixels.

Tolerance accounts for JPEG colour bleeding (default L2 < 20 per project spec).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Canonical MetalDAM colour → class map (RGB order).
METALDAM_COLOR_MAP: Dict[Tuple[int, int, int], int] = {
    (255, 0, 255): 0,   # Background / Defect
    (0, 255, 0):   1,   # Austenite
    (128, 0, 255): 2,   # Matrix
    (255, 255, 0): 3,   # Martensite-Austenite (MA)
    (255, 0, 0):   4,   # Precipitate
}

IGNORE_LABEL: int = 255


def rgb_to_class_mask(
    label_rgb: np.ndarray,
    color_map: Dict[Tuple[int, int, int], int],
    tolerance: float = 20.0,
) -> np.ndarray:
    """Convert an RGB label image to a uint8 class-index mask.

    Each pixel is matched to the nearest colour in *color_map* by L2 distance.
    Pixels whose nearest colour exceeds *tolerance* are assigned ``IGNORE_LABEL``
    (255) and excluded from loss computation via ``ignore_index=255``.

    Parameters
    ----------
    label_rgb:
        H×W×3 uint8 numpy array in RGB channel order.
    color_map:
        Mapping of ``(R, G, B)`` tuples to integer class indices.
    tolerance:
        Maximum L2 distance (in RGB space) for a pixel to be assigned a class.
        Default 20 matches the project spec for JPEG colour bleeding.

    Returns
    -------
    np.ndarray
        H×W uint8 array.  Values are class indices from *color_map* or 255.

    Raises
    ------
    ValueError
        If *label_rgb* is not a 3-channel uint8 array, or *color_map* is empty.

    Examples
    --------
    >>> from src.preprocess.label_encoding import rgb_to_class_mask, METALDAM_COLOR_MAP
    >>> import numpy as np
    >>> import cv2
    >>> label_bgr = cv2.imread("data/raw/coloured_labels/sample_01.png")
    >>> label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
    >>> mask = rgb_to_class_mask(label_rgb, METALDAM_COLOR_MAP)
    >>> print(mask.shape, mask.dtype)   # (H, W) uint8
    """
    _validate_inputs(label_rgb, color_map)

    h, w = label_rgb.shape[:2]
    n_classes = len(color_map)

    # Flatten to (N, 3) — work in float32 to avoid uint8 subtraction wrap-around.
    pixels = label_rgb.reshape(-1, 3).astype(np.float32)   # (N, 3)

    # Build (C, 3) palette and (C,) class-index arrays from the dict.
    palette = np.array(list(color_map.keys()), dtype=np.float32)   # (C, 3)
    class_ids = np.array(list(color_map.values()), dtype=np.uint8)  # (C,)

    # Squared L2 distances: broadcast (N,1,3) − (1,C,3) → (N, C).
    # Comparing squared distances avoids a sqrt and is numerically identical
    # for the argmin; we only take sqrt for the tolerance check.
    diff = pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]  # (N, C, 3)
    sq_dist = (diff ** 2).sum(axis=2)                             # (N, C)

    nearest_idx = sq_dist.argmin(axis=1)                          # (N,)
    min_sq_dist = sq_dist[np.arange(len(pixels)), nearest_idx]    # (N,)

    # Assign class or ignore label.
    mask_flat = np.where(
        min_sq_dist <= tolerance ** 2,
        class_ids[nearest_idx],
        IGNORE_LABEL,
    ).astype(np.uint8)

    n_ignored = int((mask_flat == IGNORE_LABEL).sum())
    if n_ignored > 0:
        pct = 100.0 * n_ignored / (h * w)
        logger.warning(
            "[label_encoding] %d pixels (%.2f%%) unmatched → assigned ignore "
            "label 255.  Consider raising tolerance if this is unexpectedly high.",
            n_ignored, pct,
        )

    return mask_flat.reshape(h, w)


def validate_color_map(
    label_rgb: np.ndarray,
    color_map: Dict[Tuple[int, int, int], int],
    tolerance: float = 20.0,
    min_fraction: float = 0.005,
) -> Dict[int, int]:
    """Print per-class pixel counts and warn on under-represented classes.

    Parameters
    ----------
    label_rgb:
        H×W×3 uint8 numpy array in RGB channel order.
    color_map:
        ``(R, G, B)`` → class index mapping (same as passed to
        ``rgb_to_class_mask``).
    tolerance:
        L2 tolerance forwarded to ``rgb_to_class_mask``.
    min_fraction:
        Classes covering less than this fraction of the image trigger a
        warning.  Default 0.005 = 0.5 %.

    Returns
    -------
    dict
        ``{class_index: pixel_count}`` for all classes (including 255).

    Examples
    --------
    >>> counts = validate_color_map(label_rgb, METALDAM_COLOR_MAP)
    """
    mask = rgb_to_class_mask(label_rgb, color_map, tolerance)
    total = mask.size

    class_names = _build_name_map(color_map)
    counts: Dict[int, int] = {}

    print(f"\n{'Class':>6}  {'Name':<30}  {'Pixels':>9}  {'Fraction':>8}")
    print("─" * 60)

    all_indices = sorted({*color_map.values(), IGNORE_LABEL})
    for idx in all_indices:
        count = int((mask == idx).sum())
        counts[idx] = count
        frac = count / total
        name = class_names.get(idx, "ignore" if idx == IGNORE_LABEL else "?")
        flag = ""
        if idx != IGNORE_LABEL and frac < min_fraction and count > 0:
            flag = "  ← WARNING: < 0.5 %"
            logger.warning(
                "[label_encoding] Class %d ('%s') covers only %.3f%% of image.",
                idx, name, frac * 100,
            )
        print(f"{idx:>6}  {name:<30}  {count:>9,}  {frac:>7.3%}{flag}")

    print("─" * 60)
    print(f"{'total':>6}  {'':30}  {total:>9,}  {'100.000%':>8}\n")
    return counts


# ── private helpers ──────────────────────────────────────────────────────────

def _validate_inputs(
    label_rgb: np.ndarray,
    color_map: Dict[Tuple[int, int, int], int],
) -> None:
    if label_rgb.ndim != 3 or label_rgb.shape[2] != 3:
        raise ValueError(
            f"label_rgb must be H×W×3, got shape {label_rgb.shape}."
        )
    if label_rgb.dtype != np.uint8:
        raise ValueError(
            f"label_rgb must be uint8, got {label_rgb.dtype}."
        )
    if not color_map:
        raise ValueError("color_map must contain at least one entry.")


def _build_name_map(
    color_map: Dict[Tuple[int, int, int], int],
) -> Dict[int, str]:
    """Return a best-effort {class_index: name} map from known MetalDAM classes."""
    _KNOWN: Dict[Tuple[int, int, int], str] = {
        (255, 0, 255): "Background/Defect",
        (0, 255, 0):   "Austenite",
        (128, 0, 255): "Matrix",
        (255, 255, 0): "Martensite-Austenite (MA)",
        (255, 0, 0):   "Precipitate",
    }
    return {idx: _KNOWN.get(rgb, f"class_{idx}") for rgb, idx in color_map.items()}


# ── CLI / usage example ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import cv2  # noqa: F401 — only needed in the example runner

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    # ── synthetic usage example (no real images needed) ──────────────────
    print("=== Synthetic example: 5 MetalDAM classes ===\n")

    rng = np.random.default_rng(0)
    h, w = 256, 256

    # Build a synthetic label by assigning each pixel a random class colour.
    palette_rgb = np.array(list(METALDAM_COLOR_MAP.keys()), dtype=np.uint8)  # (5, 3)
    class_ids_flat = rng.integers(0, len(palette_rgb), size=h * w)
    synthetic_rgb = palette_rgb[class_ids_flat].reshape(h, w, 3)

    # Add a little JPEG-style colour noise (within tolerance).
    noise = rng.integers(-10, 11, size=synthetic_rgb.shape, dtype=np.int16)
    synthetic_rgb = np.clip(synthetic_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mask = rgb_to_class_mask(synthetic_rgb, METALDAM_COLOR_MAP, tolerance=20)
    print(f"Input shape : {synthetic_rgb.shape}  dtype={synthetic_rgb.dtype}")
    print(f"Output shape: {mask.shape}  dtype={mask.dtype}")
    print(f"Unique values in mask: {np.unique(mask).tolist()}\n")

    validate_color_map(synthetic_rgb, METALDAM_COLOR_MAP)

    # ── real image path (optional CLI argument) ───────────────────────────
    if len(sys.argv) > 1:
        label_path = Path(sys.argv[1])
        bgr = cv2.imread(str(label_path))
        if bgr is None:
            sys.exit(f"Cannot read '{label_path}'")
        rgb = bgr[:, :, ::-1].copy()
        print(f"\n=== Real label: {label_path.name} ===")
        validate_color_map(rgb, METALDAM_COLOR_MAP)
        out_path = label_path.with_suffix(".mask.npy")
        np.save(str(out_path), rgb_to_class_mask(rgb, METALDAM_COLOR_MAP))
        print(f"Mask saved to {out_path}")

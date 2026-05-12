"""
Geometric feature extraction from segmentation masks.

All regionprops measurements are in PIXELS inside this module.
Physical values are obtained by multiplying by um_per_pixel (lengths)
or um_per_pixel² (areas) before returning — see geometric-features rule.

Outputs per image:
  - Mean equivalent grain diameter (µm)
  - Phase fraction per class
  - Mean aspect ratio (major_axis / minor_axis)
  - Boundary density (µm⁻¹): total boundary length / total area
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

logger = logging.getLogger(__name__)

IGNORE_LABEL: int = 255
N_CLASSES: int = 5

# Minimum grain area in pixels to include in statistics.
# Grains smaller than this are likely noise or boundary artefacts.
_MIN_GRAIN_PX: int = 16


def extract_geometric_features(
    mask: np.ndarray,
    um_per_pixel: float,
) -> Dict:
    """Extract grain-scale geometric features from a class-index mask.

    Parameters
    ----------
    mask:
        H×W uint8 array of class indices (0–4, 255 = ignore).
    um_per_pixel:
        Physical resolution in µm per pixel, **after** resolution
        normalisation (read from ``metadata.csv``).  Every pixel-scale
        measurement is multiplied by this value before returning.

    Returns
    -------
    dict with keys:

    ``mean_grain_diameter_um``
        Mean equivalent circular diameter across all foreground grains (µm).
        Computed as ``2 * sqrt(area / π)`` per grain.

    ``phase_fractions``
        ``{class_index: fraction}`` — fraction of non-ignore pixels belonging
        to each class.

    ``mean_aspect_ratio``
        Mean of ``major_axis_length / minor_axis_length`` per grain.
        Grains with zero minor axis (points / thin lines) are excluded.

    ``boundary_density_um``
        Total grain boundary perimeter (µm) divided by total imaged area
        (µm²), giving a boundary density in µm⁻¹.  Uses 4-connectivity
        boundary detection on the full mask.

    ``per_class_diameters_um``
        ``{class_index: [diameter_um, ...]}`` — raw diameter list per class
        for downstream statistics or histograms.

    Raises
    ------
    ValueError
        If ``um_per_pixel`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    >>> feats = extract_geometric_features(mask, um_per_pixel=0.012)
    >>> print(feats["mean_grain_diameter_um"])
    """
    if um_per_pixel <= 0:
        raise ValueError(
            f"um_per_pixel must be positive, got {um_per_pixel}."
        )

    valid_mask = mask != IGNORE_LABEL
    valid_pixels = int(valid_mask.sum())

    if valid_pixels == 0:
        logger.warning("[geometric] Mask contains only ignore pixels.")
        return _empty_result()

    # ── phase fractions ───────────────────────────────────────────────────
    phase_fractions: Dict[int, float] = {}
    for cls in range(N_CLASSES):
        cls_px = int((mask == cls).sum())
        phase_fractions[cls] = cls_px / valid_pixels

    # ── per-grain measurements ────────────────────────────────────────────
    # Label connected components per class independently so grains of the
    # same phase that touch are not merged across class boundaries.
    all_diameters_px: List[float] = []
    all_aspect_ratios: List[float] = []
    per_class_diameters_um: Dict[int, List[float]] = {c: [] for c in range(N_CLASSES)}

    for cls in range(N_CLASSES):
        binary = (mask == cls).astype(np.uint8)
        labeled = label(binary, connectivity=2)
        props = regionprops(labeled)

        for region in props:
            if region.area < _MIN_GRAIN_PX:
                continue

            # Equivalent diameter: diameter of circle with same area.
            # Still in pixels here — multiplied below.
            diam_px = 2.0 * np.sqrt(region.area / np.pi)
            all_diameters_px.append(diam_px)
            per_class_diameters_um[cls].append(diam_px * um_per_pixel)

            # Aspect ratio: skip degenerate grains (minor axis == 0).
            if region.minor_axis_length > 0:
                all_aspect_ratios.append(
                    region.major_axis_length / region.minor_axis_length
                )

    # ── convert diameters to µm ───────────────────────────────────────────
    # Multiply pixel lengths by um_per_pixel (geometric-features rule).
    diameters_um = [d * um_per_pixel for d in all_diameters_px]
    mean_diameter_um = float(np.mean(diameters_um)) if diameters_um else 0.0
    mean_aspect_ratio = float(np.mean(all_aspect_ratios)) if all_aspect_ratios else 1.0

    # ── boundary density ──────────────────────────────────────────────────
    # find_boundaries returns True for pixels adjacent to a different label.
    # Operates on the full mask so class-to-class transitions are detected.
    boundary_map = find_boundaries(mask, connectivity=1, mode="outer")
    # Ignore boundary pixels that border the ignore label.
    boundary_map = boundary_map & valid_mask

    boundary_px = int(boundary_map.sum())
    # Boundary length in µm: each pixel edge ≈ um_per_pixel.
    boundary_um = boundary_px * um_per_pixel
    # Total valid area in µm².
    area_um2 = valid_pixels * (um_per_pixel ** 2)
    boundary_density_um = boundary_um / area_um2 if area_um2 > 0 else 0.0

    logger.info(
        "[geometric] grains=%d  mean_diam=%.3f µm  mean_ar=%.3f  "
        "boundary_density=%.4f µm⁻¹",
        len(diameters_um),
        mean_diameter_um,
        mean_aspect_ratio,
        boundary_density_um,
    )

    return {
        "mean_grain_diameter_um":  mean_diameter_um,
        "phase_fractions":         phase_fractions,
        "mean_aspect_ratio":       mean_aspect_ratio,
        "boundary_density_um":     boundary_density_um,
        "per_class_diameters_um":  per_class_diameters_um,
    }


def features_to_vector(features: Dict) -> np.ndarray:
    """Flatten the feature dict to a 1-D numpy array for regression models.

    Layout (fixed order, 11 elements for 5-class MetalDAM):
        [mean_grain_diameter_um,
         phase_fraction_0, ..., phase_fraction_4,
         mean_aspect_ratio,
         boundary_density_um]

    Parameters
    ----------
    features:
        Dict returned by :func:`extract_geometric_features`.

    Returns
    -------
    np.ndarray
        Shape ``(11,)``, dtype float64.
    """
    pf = features["phase_fractions"]
    return np.array(
        [features["mean_grain_diameter_um"]]
        + [pf.get(c, 0.0) for c in range(N_CLASSES)]
        + [features["mean_aspect_ratio"], features["boundary_density_um"]],
        dtype=np.float64,
    )


# ── private helpers ───────────────────────────────────────────────────────────

def _empty_result() -> Dict:
    return {
        "mean_grain_diameter_um":  0.0,
        "phase_fractions":         {c: 0.0 for c in range(N_CLASSES)},
        "mean_aspect_ratio":       1.0,
        "boundary_density_um":     0.0,
        "per_class_diameters_um":  {c: [] for c in range(N_CLASSES)},
    }


# ── usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    rng = np.random.default_rng(42)
    h, w = 512, 512
    mock_mask = rng.integers(0, N_CLASSES, size=(h, w), dtype=np.uint8)
    # Sprinkle some ignore pixels along the border.
    mock_mask[:4, :] = IGNORE_LABEL
    mock_mask[-4:, :] = IGNORE_LABEL

    um_per_pixel = 0.012

    features = extract_geometric_features(mock_mask, um_per_pixel=um_per_pixel)

    print("=== Geometric Feature Report ===\n")
    print(f"Mean grain diameter  : {features['mean_grain_diameter_um']:.4f} µm")
    print(f"Mean aspect ratio    : {features['mean_aspect_ratio']:.4f}")
    print(f"Boundary density     : {features['boundary_density_um']:.6f} µm⁻¹")
    print("\nPhase fractions:")
    _CLASS_NAMES = {0: "Background", 1: "Austenite", 2: "Matrix",
                    3: "Martensite-Austenite", 4: "Precipitate"}
    for cls, frac in features["phase_fractions"].items():
        print(f"  Class {cls} ({_CLASS_NAMES.get(cls, '?'):<22}): {frac:.4%}")

    vec = features_to_vector(features)
    print(f"\nFeature vector (shape={vec.shape}): {np.round(vec, 5).tolist()}")

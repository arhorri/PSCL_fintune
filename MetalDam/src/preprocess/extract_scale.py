"""
Node 01 — µm/pixel recovery from SEM micrographs.

Priority:
  1. CSV metadata lookup (fastest, most reliable).
  2. Image-based scale-bar detection (white bar in bottom-left info strip).
  3. OCR of the scale label to auto-detect known_length_um (optional).

Pipeline rule: this module must run BEFORE crop_align.py on every image.
The info bar is the only source of µm/pixel ground truth.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
# Fraction of image height reserved for the SEM info bar at the bottom.
_INFO_BAR_FRACTION = 0.12
# Fraction of image width to inspect for the scale bar (left side of info bar).
_SCALE_REGION_FRACTION = 0.35
# Brightness threshold for isolating the white scale bar (0–255).
_BAR_THRESHOLD = 200
# Minimum horizontal run of white pixels to count as the scale bar.
_MIN_BAR_PX = 10
# OCR scale patterns, e.g. "1 µm", "500 nm", "0.5 mm"
_SCALE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(nm|µm|um|μm|mm)", re.IGNORECASE
)
_UM_FACTORS: dict[str, float] = {
    "nm": 1e-3,
    "µm": 1.0, "um": 1.0, "μm": 1.0,
    "mm": 1e3,
}


# ── public API ───────────────────────────────────────────────────────────────

def extract_um_per_pixel(
    img_path: str | Path,
    csv_path: str | Path | None = None,
    known_length_um: float = 1.0,
    *,
    use_ocr: bool = True,
) -> float:
    """Return µm per pixel for a SEM micrograph.

    Resolution is determined by the first successful strategy:

    1. **CSV lookup** — reads ``um_per_pixel`` from *csv_path* where
       ``filename`` matches *img_path* stem.
    2. **OCR** (when *use_ocr* is True) — crops the info bar, runs
       ``pytesseract`` to parse a scale label such as ``"1 µm"`` and
       auto-sets *known_length_um*.
    3. **Scale-bar detection** — thresholds the bottom-left info strip to
       find the white horizontal bar; returns
       ``known_length_um / bar_pixels``.

    Parameters
    ----------
    img_path:
        Path to the original SEM image (info bar must still be present).
    csv_path:
        Optional path to ``metadata.csv``.  Must contain columns
        ``filename`` (image stem) and ``um_per_pixel``.
    known_length_um:
        Physical length the scale bar represents, in µm.  Used as
        fallback when OCR is disabled or fails.
    use_ocr:
        Whether to attempt pytesseract OCR on the info bar.

    Returns
    -------
    float
        µm per pixel (positive, > 0).

    Raises
    ------
    RuntimeError
        If no strategy succeeds.

    Examples
    --------
    >>> um_px = extract_um_per_pixel(
    ...     "data/raw/images/sample_01.jpg",
    ...     csv_path="metadata.csv",
    ...     known_length_um=1.0,
    ...     use_ocr=True,
    ... )
    >>> print(f"{um_px:.6f} µm/px")
    """
    img_path = Path(img_path)

    # ── strategy 1: CSV lookup ────────────────────────────────────────────
    if csv_path is not None:
        result = _lookup_csv(img_path, Path(csv_path))
        if result is not None:
            logger.info("[extract_scale] %s — method: CSV (%.6f µm/px)",
                        img_path.name, result)
            return result

    img = _load_gray(img_path)
    info_strip = _crop_info_strip(img)

    # ── strategy 2: OCR ──────────────────────────────────────────────────
    if use_ocr:
        ocr_um = _ocr_scale_length(info_strip)
        if ocr_um is not None:
            known_length_um = ocr_um
            logger.debug("[extract_scale] OCR detected scale label: %.4f µm",
                         known_length_um)

    # ── strategy 3: pixel measurement ────────────────────────────────────
    bar_pixels = _measure_bar_pixels(info_strip)
    if bar_pixels > 0:
        result = known_length_um / bar_pixels
        method = "OCR+bar" if use_ocr else "bar"
        logger.info("[extract_scale] %s — method: %s (bar=%d px → %.6f µm/px)",
                    img_path.name, method, bar_pixels, result)
        return result

    raise RuntimeError(
        f"extract_um_per_pixel: could not determine scale for '{img_path}'. "
        "Check that the info bar is present and the image is not pre-cropped."
    )


# ── private helpers ──────────────────────────────────────────────────────────

def _lookup_csv(img_path: Path, csv_path: Path) -> float | None:
    """Return um_per_pixel from CSV, or None if not found / invalid."""
    if not csv_path.exists():
        logger.debug("[extract_scale] CSV not found: %s", csv_path)
        return None

    try:
        df = pd.read_csv(csv_path, dtype={"filename": str})
    except Exception as exc:
        logger.warning("[extract_scale] Could not read CSV: %s", exc)
        return None

    if "filename" not in df.columns or "um_per_pixel" not in df.columns:
        logger.warning("[extract_scale] CSV missing required columns "
                       "(filename, um_per_pixel).")
        return None

    stem = img_path.stem
    # Match on stem without extension so .jpg / .png variants both hit.
    row = df[df["filename"].str.replace(r"\.\w+$", "", regex=True) == stem]
    if row.empty:
        logger.debug("[extract_scale] '%s' not found in CSV.", stem)
        return None

    value = row.iloc[0]["um_per_pixel"]
    if pd.isna(value) or float(value) <= 0:
        logger.warning("[extract_scale] CSV row for '%s' has invalid "
                       "um_per_pixel: %s", stem, value)
        return None

    return float(value)


def _load_gray(img_path: Path) -> np.ndarray:
    """Load image as uint8 grayscale; raise if not found."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: '{img_path}'")
    return img


def _crop_info_strip(gray: np.ndarray) -> np.ndarray:
    """Return the bottom-left region that contains the SEM info bar."""
    h, w = gray.shape
    bar_h = max(1, int(h * _INFO_BAR_FRACTION))
    bar_w = max(1, int(w * _SCALE_REGION_FRACTION))
    return gray[h - bar_h:h, 0:bar_w]


def _measure_bar_pixels(strip: np.ndarray) -> int:
    """Return the pixel length of the longest white horizontal run in *strip*.

    The scale bar in SEM info strips is a solid bright horizontal line.
    We threshold, find contiguous white rows, and return the width of the
    widest connected component that passes the minimum-length guard.
    """
    _, binary = cv2.threshold(strip, _BAR_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Connected-component analysis is more robust than per-row runs when the
    # bar has tiny gaps due to JPEG compression.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    best = 0
    for i in range(1, num_labels):          # skip background label 0
        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Scale bars are wide and thin.
        if comp_w >= _MIN_BAR_PX and comp_w > comp_h * 3:
            best = max(best, comp_w)

    return best


def _ocr_scale_length(strip: np.ndarray) -> float | None:
    """Parse the scale label from the info strip via pytesseract.

    Returns the length in µm, or None if parsing fails.
    """
    try:
        import pytesseract
    except ImportError:
        logger.debug("[extract_scale] pytesseract not installed; skipping OCR.")
        return None

    # Upscale to improve OCR accuracy on small info bars.
    upscaled = cv2.resize(strip, None, fx=3, fy=3,
                          interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(
        upscaled,
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,µumnmMk ",
    )
    logger.debug("[extract_scale] OCR raw text: %r", text)

    match = _SCALE_PATTERN.search(text)
    if match is None:
        logger.debug("[extract_scale] No scale pattern found in OCR output.")
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    return value * _UM_FACTORS.get(unit, 1.0)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 01: extract µm/pixel for one or all SEM images."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--image", default=None,
                        help="Single image path (overrides config data dir).")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    csv_path = Path("metadata.csv")
    known_um = float(cfg.get("known_scale_bar_um", 1.0))

    if args.image:
        paths = [Path(args.image)]
    else:
        raw_dir = Path(cfg.get("raw_image_dir", "data/raw/images"))
        paths = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png"))

    results: list[dict] = []
    for p in paths:
        try:
            um_px = extract_um_per_pixel(p, csv_path, known_um)
            results.append({"filename": p.stem, "um_per_pixel": um_px})
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("%s", exc)

    if results:
        out_df = pd.DataFrame(results)
        print(out_df.to_string(index=False))

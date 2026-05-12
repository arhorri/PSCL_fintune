"""
Node 04 — Orphan detection and image/label shape validation.

Scans images_dir and labels_dir, matches pairs by stem, and reports:
  - Images with no matching label (orphaned image).
  - Labels with no matching image (orphaned label).
  - Matched pairs whose H×W dimensions disagree.

Raises RuntimeError listing every failure so the pipeline never proceeds
on corrupt data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import pandas as pd

logger = logging.getLogger(__name__)

_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_LABEL_EXTS: Set[str] = {".png"}


def validate_pairs(
    images_dir: str | Path,
    labels_dir: str | Path,
) -> pd.DataFrame:
    """Validate that every SEM image has a matching, shape-compatible label.

    Matching is done on file stem (case-sensitive, extension-agnostic).
    A pair is valid when both files exist and share the same H×W dimensions.

    Parameters
    ----------
    images_dir:
        Directory containing cropped SEM micrographs (output of Node 02).
    labels_dir:
        Directory containing coloured label PNGs (bar-free originals).

    Returns
    -------
    pd.DataFrame
        One row per unique stem with columns:
        ``filename``, ``has_image``, ``has_label``, ``shapes_match``, ``notes``.
        ``shapes_match`` is ``None`` for orphaned files (no pair to compare).

    Raises
    ------
    RuntimeError
        If any orphaned file or shape mismatch is found.  The message lists
        every failure so all issues can be fixed in one pass.

    Examples
    --------
    >>> from src.preprocess.validate_pairs import validate_pairs
    >>> df = validate_pairs(
    ...     "data/processed/images",
    ...     "data/raw/coloured_labels",
    ... )
    >>> print(df)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    _require_dir(images_dir, "images_dir")
    _require_dir(labels_dir, "labels_dir")

    img_map = _index_dir(images_dir, _IMAGE_EXTS)
    lbl_map = _index_dir(labels_dir, _LABEL_EXTS)

    all_stems = sorted(img_map.keys() | lbl_map.keys())
    if not all_stems:
        raise RuntimeError(
            f"No files found in '{images_dir}' or '{labels_dir}'."
        )

    rows: List[Dict] = []
    failures: List[str] = []

    for stem in all_stems:
        img_path = img_map.get(stem)
        lbl_path = lbl_map.get(stem)
        has_image = img_path is not None
        has_label = lbl_path is not None

        if not has_image:
            note = f"orphaned label — no image for '{stem}'"
            rows.append(_row(stem, has_image, has_label, None, note))
            failures.append(f"  [orphan-label]  {lbl_path}")
            continue

        if not has_label:
            note = f"orphaned image — no label for '{stem}'"
            rows.append(_row(stem, has_image, has_label, None, note))
            failures.append(f"  [orphan-image]  {img_path}")
            continue

        shapes_match, note = _check_shapes(img_path, lbl_path)
        rows.append(_row(stem, has_image, has_label, shapes_match, note))
        if not shapes_match:
            failures.append(f"  [shape-mismatch]  {stem}: {note}")

    df = pd.DataFrame(
        rows,
        columns=["filename", "has_image", "has_label", "shapes_match", "notes"],
    )

    _print_summary(df, images_dir, labels_dir)

    if failures:
        failure_block = "\n".join(failures)
        raise RuntimeError(
            f"validate_pairs found {len(failures)} failure(s):\n{failure_block}"
        )

    return df


# ── private helpers ──────────────────────────────────────────────────────────

def _require_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: '{path}'")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: '{path}'")


def _index_dir(directory: Path, exts: Set[str]) -> Dict[str, Path]:
    """Return {stem: path} for all matching files in *directory*."""
    index: Dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in exts:
            if p.stem in index:
                logger.warning(
                    "[validate_pairs] Duplicate stem '%s' in '%s'; "
                    "keeping first match (%s).",
                    p.stem, directory, index[p.stem].name,
                )
            else:
                index[p.stem] = p
    return index


def _check_shapes(
    img_path: Path, lbl_path: Path
) -> Tuple[bool, str]:
    """Load both files, compare H×W.  Return (match, note)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, f"cannot read image '{img_path.name}'"

    lbl = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
    if lbl is None:
        return False, f"cannot read label '{lbl_path.name}'"

    ih, iw = img.shape[:2]
    lh, lw = lbl.shape[:2]

    if (ih, iw) == (lh, lw):
        return True, f"ok — {ih}×{iw}"

    return False, (
        f"image {ih}×{iw} vs label {lh}×{lw} — "
        "run crop_align.py to fix image height"
    )


def _row(
    stem: str,
    has_image: bool,
    has_label: bool,
    shapes_match: bool | None,
    note: str,
) -> Dict:
    return {
        "filename": stem,
        "has_image": has_image,
        "has_label": has_label,
        "shapes_match": shapes_match,
        "notes": note,
    }


def _print_summary(df: pd.DataFrame, images_dir: Path, labels_dir: Path) -> None:
    total = len(df)
    valid = int(df["shapes_match"].eq(True).sum())
    orphan_img = int(df["has_image"].eq(True).eq(df["has_label"].eq(False)).sum())
    orphan_lbl = int(df["has_label"].eq(True).eq(df["has_image"].eq(False)).sum())
    mismatch = int(df["shapes_match"].eq(False).sum())

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Pair validation report")
    print(f"  images : {images_dir}")
    print(f"  labels : {labels_dir}")
    print(sep)
    print(f"  Total stems examined   : {total:>5}")
    print(f"  Valid matched pairs    : {valid:>5}")
    print(f"  Orphaned images        : {orphan_img:>5}")
    print(f"  Orphaned labels        : {orphan_lbl:>5}")
    print(f"  Shape mismatches       : {mismatch:>5}")
    print(sep)

    problem_df = df[df["shapes_match"].ne(True)]
    if not problem_df.empty:
        print("\n  Problem rows:\n")
        print(
            problem_df[["filename", "has_image", "has_label", "shapes_match", "notes"]]
            .to_string(index=False)
        )
        print()
    else:
        print("\n  All pairs are valid.\n")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 04: validate image/label pairs."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument(
        "--images-dir", default=None,
        help="Override processed images directory from config.",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="Override raw labels directory from config.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    images_dir = args.images_dir or cfg.get(
        "processed_image_dir", "data/processed/images"
    )
    labels_dir = args.labels_dir or cfg.get(
        "raw_label_dir", "data/raw/coloured_labels"
    )

    try:
        df = validate_pairs(images_dir, labels_dir)
        csv_out = Path("metadata_validation.csv")
        df.to_csv(csv_out, index=False)
        logger.info("[validate_pairs] Report saved to %s", csv_out)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)

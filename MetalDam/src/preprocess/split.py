"""
Node 08 — Parent-level stratified train / val / test split.

CRITICAL: splitting must happen at the PARENT IMAGE level.  All patches that
share a parent_id receive the same split label.  Splitting at patch level
leaks spatial correlations between train and val and inflates metrics.

If the metadata CSV contains patch-level rows (parent_id column present),
this module aggregates to unique parents, stratifies, then propagates the
split assignment back to every child row.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_dataset(
    metadata_csv: str | Path,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    stratify_col: str = "dominant_class",
) -> pd.DataFrame:
    """Add a 'split' column to metadata.csv with parent-level stratification.

    Workflow
    --------
    1. Load ``metadata_csv``.
    2. Derive ``dominant_class`` per parent from ``class_histogram`` if the
       column is absent (or contains nulls).
    3. Aggregate to one row per parent image.
    4. Stratified split by *stratify_col* → train / val / test.
    5. Fall back to unstratified split if any stratum has too few samples.
    6. Propagate split assignments to every child patch row.
    7. Save the updated CSV (never deletes existing columns).
    8. Print per-split class distribution for verification.

    Parameters
    ----------
    metadata_csv:
        Path to ``metadata.csv`` (may contain parent- or patch-level rows).
    train:
        Fraction of parents assigned to the training split.
    val:
        Fraction of parents assigned to the validation split.
    test:
        Fraction of parents assigned to the test split.  Must satisfy
        ``train + val + test == 1.0`` within floating-point tolerance.
    seed:
        Random seed for reproducibility.
    stratify_col:
        Column used for stratification (must be categorical / integer after
        derivation).  Default is ``'dominant_class'``.

    Returns
    -------
    pd.DataFrame
        Full metadata (patch-level) with ``split`` column filled in.

    Raises
    ------
    FileNotFoundError
        If *metadata_csv* does not exist.
    ValueError
        If ratios do not sum to 1, if no parent IDs are found, or if
        *stratify_col* cannot be derived.

    Examples
    --------
    >>> from src.preprocess.split import split_dataset
    >>> df = split_dataset("metadata.csv", train=0.70, val=0.15, test=0.15)
    >>> print(df["split"].value_counts())
    """
    meta_path = Path(metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: '{meta_path}'")

    _validate_ratios(train, val, test)

    df = pd.read_csv(meta_path)
    _require_columns(df, ["filename"])

    # ── resolve parent identifier ─────────────────────────────────────────
    # If patch-level rows exist, group by parent_id; otherwise treat each
    # row as its own parent.
    if "parent_id" in df.columns:
        df["parent_id"] = df["parent_id"].fillna(df["filename"].apply(
            lambda f: Path(str(f)).stem
        ))
    else:
        df["parent_id"] = df["filename"].apply(lambda f: Path(str(f)).stem)

    # ── derive dominant_class if missing ──────────────────────────────────
    if stratify_col == "dominant_class":
        df = _ensure_dominant_class(df)

    if stratify_col not in df.columns:
        raise ValueError(
            f"stratify_col='{stratify_col}' not found in CSV after derivation. "
            f"Available columns: {list(df.columns)}"
        )

    # ── aggregate to parent level ─────────────────────────────────────────
    parent_df = (
        df.groupby("parent_id", sort=True)[stratify_col]
        .agg(_majority_vote)
        .reset_index()
        .rename(columns={stratify_col: "strat_label"})
    )
    parent_df = parent_df.sort_values("parent_id").reset_index(drop=True)

    if len(parent_df) == 0:
        raise ValueError("No parent IDs found in metadata.")

    logger.info(
        "[split] %d unique parents found.  Stratifying by '%s'.",
        len(parent_df), stratify_col,
    )

    # ── stratified split ──────────────────────────────────────────────────
    assignments = _stratified_split(parent_df, train, val, test, seed)

    # ── propagate to all child rows ───────────────────────────────────────
    df["split"] = df["parent_id"].map(assignments)
    n_unassigned = df["split"].isna().sum()
    if n_unassigned:
        logger.warning(
            "[split] %d rows could not be assigned a split; defaulting to 'train'.",
            n_unassigned,
        )
        df["split"] = df["split"].fillna("train")

    df.to_csv(meta_path, index=False)
    logger.info("[split] Saved updated metadata to '%s'.", meta_path)

    _print_distribution(df, stratify_col)

    return df


# ── private helpers ──────────────────────────────────────────────────────────

def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train + val + test must equal 1.0, got {train}+{val}+{test}={total:.6f}."
        )
    for name, v in [("train", train), ("val", val), ("test", test)]:
        if v <= 0:
            raise ValueError(f"Split ratio '{name}' must be > 0, got {v}.")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")


def _ensure_dominant_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'dominant_class' derived from 'class_histogram' if not present."""
    if "dominant_class" in df.columns and df["dominant_class"].notna().all():
        return df

    if "class_histogram" not in df.columns:
        raise ValueError(
            "'dominant_class' column is absent and cannot be derived: "
            "'class_histogram' column is also missing."
        )

    def _dominant(hist_json: str) -> int:
        try:
            hist: Dict[str, int] = json.loads(hist_json)
        except (TypeError, json.JSONDecodeError):
            return -1
        if not hist:
            return -1
        # Exclude ignore label (255) from dominance calculation.
        filtered = {k: v for k, v in hist.items() if int(k) != 255}
        if not filtered:
            return -1
        return int(max(filtered, key=filtered.__getitem__))

    df["dominant_class"] = df["class_histogram"].apply(_dominant)
    n_unknown = int((df["dominant_class"] == -1).sum())
    if n_unknown:
        logger.warning(
            "[split] %d rows have unparseable class_histogram; "
            "dominant_class set to -1 (will be treated as own stratum).",
            n_unknown,
        )
    return df


def _majority_vote(series: pd.Series) -> int:
    """Return the most frequent value in a series (parent-level aggregation)."""
    return int(series.mode().iloc[0])


def _stratified_split(
    parent_df: pd.DataFrame,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> Dict[str, str]:
    """Return {parent_id: split_name} for all parents."""
    parents = parent_df["parent_id"].values
    labels  = parent_df["strat_label"].values

    # sklearn needs at least 2 samples per stratum.
    stratum_counts = pd.Series(labels).value_counts()
    min_count = stratum_counts.min()
    use_stratify = min_count >= 2

    if not use_stratify:
        logger.warning(
            "[split] Some strata have < 2 samples — falling back to "
            "unstratified split."
        )
        stratify_arg = None
    else:
        stratify_arg = labels

    # First cut: train vs (val + test).
    val_test_frac = val + test
    train_ids, valtest_ids, _, valtest_labels = train_test_split(
        parents,
        labels,
        test_size=val_test_frac,
        random_state=seed,
        stratify=stratify_arg,
    )

    # Second cut: val vs test inside the (val + test) pool.
    test_frac_of_valtest = test / val_test_frac

    if len(valtest_ids) < 2:
        # Edge case: too few samples to split further.
        val_ids  = valtest_ids
        test_ids = np.array([], dtype=valtest_ids.dtype)
    else:
        valtest_stratify = (
            valtest_labels
            if use_stratify and pd.Series(valtest_labels).value_counts().min() >= 2
            else None
        )
        val_ids, test_ids = train_test_split(
            valtest_ids,
            test_size=test_frac_of_valtest,
            random_state=seed,
            stratify=valtest_stratify,
        )

    assignments: Dict[str, str] = {}
    for pid in train_ids:
        assignments[pid] = "train"
    for pid in val_ids:
        assignments[pid] = "val"
    for pid in test_ids:
        assignments[pid] = "test"

    logger.info(
        "[split] Parent counts — train: %d  val: %d  test: %d",
        len(train_ids), len(val_ids), len(test_ids),
    )
    return assignments


def _print_distribution(df: pd.DataFrame, stratify_col: str) -> None:
    splits = ("train", "val", "test")
    has_strat = stratify_col in df.columns

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  Split distribution  (stratify_col='{stratify_col}')")
    print(sep)

    # Patch counts per split.
    split_counts = df["split"].value_counts().reindex(splits, fill_value=0)
    parent_counts = (
        df.groupby("split")["parent_id"].nunique().reindex(splits, fill_value=0)
        if "parent_id" in df.columns
        else pd.Series(0, index=splits)
    )
    total_patches = len(df)

    print(f"  {'Split':<8}  {'Parents':>8}  {'Patches':>9}  {'% Patches':>10}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*10}")
    for s in splits:
        pct = 100.0 * split_counts[s] / total_patches if total_patches else 0
        print(f"  {s:<8}  {parent_counts[s]:>8}  {split_counts[s]:>9,}  {pct:>9.1f}%")
    print(sep)

    # Per-split class distribution (dominant class over patches).
    if has_strat:
        print(f"\n  Class distribution per split (by dominant_class):\n")
        pivot = (
            df.groupby(["split", stratify_col])
            .size()
            .unstack(fill_value=0)
            .reindex(splits, fill_value=0)
        )
        print(pivot.to_string())
        print()

    print(sep + "\n")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Node 08: stratified parent-level train/val/test split."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    split_dataset(
        metadata_csv=cfg.get("metadata_csv", "metadata.csv"),
        train=float(cfg.get("train_ratio", 0.70)),
        val=float(cfg.get("val_ratio",   0.15)),
        test=float(cfg.get("test_ratio",  0.15)),
        seed=int(cfg.get("seed", 42)),
    )

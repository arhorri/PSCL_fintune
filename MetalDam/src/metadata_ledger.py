"""
MetadataLedger — single-writer interface for metadata.csv.

Every preprocessing node appends rows through this class so that:
  - Column schema is enforced in one place.
  - Writes are atomic (tmp-file + rename — no partial writes on crash).
  - The CSV never has missing um_per_pixel values (validated on add).

Schema (from metadata-schema.md):
    filename        str   — patch filename
    parent_id       str   — stem of the source SEM image
    split           str   — train | val | test  (nullable until Node 08)
    um_per_pixel    float — µm/px after resolution normalisation
    magnification_kx float — from SEM info bar (nullable if OCR failed)
    kV              float — accelerating voltage (nullable if OCR failed)
    height          int   — patch height px
    width           int   — patch width px
    patch_y         int   — top-left y in parent image
    patch_x         int   — top-left x in parent image
    class_histogram str   — JSON {"0": n, "1": n, ...}
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Columns that MUST be present and non-null in every row.
_REQUIRED: List[str] = [
    "filename",
    "parent_id",
    "um_per_pixel",
    "height",
    "width",
    "patch_y",
    "patch_x",
    "class_histogram",
]

# Full ordered schema — all columns the CSV will ever contain.
_SCHEMA: Dict[str, Any] = {
    "filename":         pd.StringDtype(),
    "parent_id":        pd.StringDtype(),
    "split":            pd.StringDtype(),
    "um_per_pixel":     "float64",
    "magnification_kx": "float64",
    "kV":               "float64",
    "height":           "Int64",
    "width":            "Int64",
    "patch_y":          "Int64",
    "patch_x":          "Int64",
    "class_histogram":  pd.StringDtype(),
}

_VALID_SPLITS = {"train", "val", "test"}


class MetadataLedger:
    """Append-only ledger for patch metadata, backed by a CSV file.

    Parameters
    ----------
    csv_path:
        Path to ``metadata.csv``.  Loaded if it exists; a new empty
        DataFrame is created otherwise.

    Examples
    --------
    >>> ledger = MetadataLedger("metadata.csv")
    >>> ledger.add_entry(
    ...     filename="sample_01__y00000_x00000.png",
    ...     parent_id="sample_01",
    ...     um_per_pixel=0.012,
    ...     height=256, width=256,
    ...     patch_y=0, patch_x=0,
    ...     class_histogram={"1": 40000, "2": 25536},
    ... )
    >>> ledger.save()
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._path = Path(csv_path)
        self._df = self._load()

    # ── public API ────────────────────────────────────────────────────────

    def add_entry(self, **kwargs: Any) -> None:
        """Append one row to the ledger.

        Parameters
        ----------
        **kwargs:
            Column name → value mappings.  All ``_REQUIRED`` columns must
            be present and non-None.  ``class_histogram`` may be passed as
            a ``dict`` and will be serialised to JSON automatically.
            Unknown columns are accepted and appended as-is (new columns
            are added to the schema without breaking existing rows).

        Raises
        ------
        KeyError
            If any required column is missing.
        ValueError
            If ``um_per_pixel`` is non-positive, ``split`` is not one of
            ``train | val | test`` (when provided), or ``class_histogram``
            is not serialisable.
        """
        kwargs = self._coerce(kwargs)
        self._validate(kwargs)
        row = pd.DataFrame([kwargs])
        self._df = (
            pd.concat([self._df, row], ignore_index=True)
            if not self._df.empty
            else row
        )

    def add_entries(self, rows: List[Dict[str, Any]]) -> None:
        """Append multiple rows in one call.

        Equivalent to calling :meth:`add_entry` in a loop but batches the
        DataFrame concatenation for better performance on large patch sets.
        """
        if not rows:
            return
        coerced = [self._coerce(r) for r in rows]
        for r in coerced:
            self._validate(r)
        batch = pd.DataFrame(coerced)
        self._df = (
            pd.concat([self._df, batch], ignore_index=True)
            if not self._df.empty
            else batch
        )

    def get_split(self, name: str) -> pd.DataFrame:
        """Return a copy of all rows belonging to *name* split.

        Parameters
        ----------
        name:
            One of ``'train'``, ``'val'``, ``'test'``.

        Returns
        -------
        pd.DataFrame
            Filtered view.  Empty DataFrame if the split has no rows yet.

        Raises
        ------
        ValueError
            If *name* is not a valid split label.
        """
        if name not in _VALID_SPLITS:
            raise ValueError(
                f"Unknown split '{name}'. Choose from {sorted(_VALID_SPLITS)}."
            )
        if "split" not in self._df.columns or self._df.empty:
            return pd.DataFrame(columns=list(_SCHEMA))
        return self._df[self._df["split"] == name].copy()

    def update_split_assignments(self, assignments: Dict[str, str]) -> None:
        """Bulk-set the 'split' column from a {parent_id: split} mapping.

        Called by Node 08 (split.py) after stratified assignment.
        Patches inherit their parent's split.

        Parameters
        ----------
        assignments:
            ``{parent_id: "train" | "val" | "test"}`` for all parents.
        """
        invalid = {v for v in assignments.values() if v not in _VALID_SPLITS}
        if invalid:
            raise ValueError(f"Invalid split labels in assignments: {invalid}")
        self._df["split"] = self._df["parent_id"].map(assignments)

    def save(self) -> None:
        """Atomically write the ledger to disk (tmp → rename).

        The rename is atomic on POSIX systems.  On Windows, ``os.replace``
        is used which is as close to atomic as Windows allows.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp", prefix=".metadata_"
        )
        try:
            with os.fdopen(fd, "w", newline="") as fh:
                self._df.to_csv(fh, index=False)
            os.replace(tmp_path, self._path)
        except Exception:
            # Clean up the temp file if anything goes wrong.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def summary(self) -> None:
        """Print a human-readable summary of the ledger contents."""
        df = self._df
        total = len(df)

        sep = "─" * 66
        print(f"\n{sep}")
        print(f"  MetadataLedger summary  ({self._path})")
        print(f"  Total rows : {total:,}")
        print(sep)

        if total == 0:
            print("  (empty)\n" + sep + "\n")
            return

        # ── per-split counts ──────────────────────────────────────────────
        if "split" in df.columns:
            print(f"\n  {'Split':<8}  {'Patches':>9}  {'Parents':>9}  {'mean µm/px':>12}")
            print(f"  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*12}")
            for split in ("train", "val", "test", None):
                mask = (
                    df["split"] == split if split is not None
                    else df["split"].isna()
                )
                sub = df[mask]
                if sub.empty:
                    continue
                n_parents = sub["parent_id"].nunique() if "parent_id" in df.columns else "—"
                mean_um = (
                    f"{sub['um_per_pixel'].mean():.6f}"
                    if "um_per_pixel" in df.columns
                    else "—"
                )
                label = split if split is not None else "(unassigned)"
                print(f"  {label:<8}  {len(sub):>9,}  {n_parents!s:>9}  {mean_um:>12}")

        # ── dataset-wide um_per_pixel ─────────────────────────────────────
        if "um_per_pixel" in df.columns:
            um = df["um_per_pixel"].dropna()
            print(f"\n  µm/pixel — mean: {um.mean():.6f}  "
                  f"std: {um.std():.6f}  "
                  f"min: {um.min():.6f}  "
                  f"max: {um.max():.6f}")

        # ── dataset-wide class distribution ───────────────────────────────
        if "class_histogram" in df.columns:
            totals: Dict[str, int] = {}
            for entry in df["class_histogram"].dropna():
                try:
                    hist = json.loads(entry) if isinstance(entry, str) else entry
                    for cls, cnt in hist.items():
                        totals[str(cls)] = totals.get(str(cls), 0) + int(cnt)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

            if totals:
                grand = sum(totals.values())
                print(f"\n  Class distribution (all splits):")
                print(f"  {'Class':>6}  {'Pixels':>12}  {'Fraction':>9}")
                print(f"  {'─'*6}  {'─'*12}  {'─'*9}")
                for cls in sorted(totals, key=lambda k: int(k) if k.isdigit() else 999):
                    cnt = totals[cls]
                    print(f"  {cls:>6}  {cnt:>12,}  {cnt/grand:>8.3%}")

        print(f"\n{sep}\n")

    # ── internals ─────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=list(_SCHEMA))
        df = pd.read_csv(self._path, dtype=str)          # read all as str first
        df = df.reindex(
            columns=list(_SCHEMA) + [c for c in df.columns if c not in _SCHEMA]
        )
        # Cast known columns to their declared types, tolerating missing ones.
        for col, dtype in _SCHEMA.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass
        return df

    @staticmethod
    def _coerce(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise class_histogram to a JSON string."""
        out = dict(kwargs)
        if "class_histogram" in out and isinstance(out["class_histogram"], dict):
            out["class_histogram"] = json.dumps(
                {str(k): int(v) for k, v in out["class_histogram"].items()}
            )
        return out

    @staticmethod
    def _validate(kwargs: Dict[str, Any]) -> None:
        # Required fields must be present and non-None / non-NaN.
        missing = [k for k in _REQUIRED if kwargs.get(k) is None]
        if missing:
            raise KeyError(
                f"add_entry: missing required fields: {missing}"
            )

        # um_per_pixel must be a positive finite number.
        um = kwargs.get("um_per_pixel")
        try:
            if float(um) <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(
                f"um_per_pixel must be a positive float, got {um!r}."
            )

        # Validate split label when provided.
        split = kwargs.get("split")
        if split is not None and split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}."
            )

        # Validate class_histogram is valid JSON.
        ch = kwargs.get("class_histogram")
        if isinstance(ch, str):
            try:
                json.loads(ch)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"class_histogram is not valid JSON: {exc}"
                )


# ── usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil

    demo_path = Path("/tmp/demo_metadata.csv")
    demo_path.unlink(missing_ok=True)

    ledger = MetadataLedger(demo_path)

    # Add individual entries (as a preprocessing node would).
    ledger.add_entry(
        filename="sample_01__y00000_x00000.png",
        parent_id="sample_01",
        um_per_pixel=0.012,
        magnification_kx=5.0,
        kV=20.0,
        height=256,
        width=256,
        patch_y=0,
        patch_x=0,
        class_histogram={"0": 4096, "1": 28672, "2": 32768},
    )
    ledger.add_entry(
        filename="sample_01__y00000_x00128.png",
        parent_id="sample_01",
        um_per_pixel=0.012,
        magnification_kx=5.0,
        kV=20.0,
        height=256,
        width=256,
        patch_y=0,
        patch_x=128,
        class_histogram={"1": 20000, "3": 45536},
    )

    # Batch add (as patch_extract.py would use).
    batch = [
        dict(
            filename=f"sample_02__y{y:05d}_x00000.png",
            parent_id="sample_02",
            um_per_pixel=0.015,
            height=256,
            width=256,
            patch_y=y,
            patch_x=0,
            class_histogram={"2": 65536},
        )
        for y in range(0, 512, 256)
    ]
    ledger.add_entries(batch)

    # Simulate Node 08 split assignment.
    ledger.update_split_assignments({
        "sample_01": "train",
        "sample_02": "val",
    })

    # Atomic save.
    ledger.save()
    print(f"Saved to {demo_path}\n")

    # Reload and query.
    ledger2 = MetadataLedger(demo_path)
    ledger2.summary()

    train_df = ledger2.get_split("train")
    print(f"Train rows: {len(train_df)}")
    print(train_df[["filename", "parent_id", "um_per_pixel", "split"]].to_string(index=False))

    # Test missing required field raises KeyError.
    try:
        ledger2.add_entry(filename="orphan.png")
    except KeyError as exc:
        print(f"\nExpected error: {exc}")

    # Test bad um_per_pixel raises ValueError.
    try:
        ledger2.add_entry(
            filename="bad.png", parent_id="p", um_per_pixel=-1,
            height=256, width=256, patch_y=0, patch_x=0,
            class_histogram={},
        )
    except ValueError as exc:
        print(f"Expected error: {exc}")

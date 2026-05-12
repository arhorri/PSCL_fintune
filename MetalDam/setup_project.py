"""
Creates the full MetalDAM project directory tree, stub __init__.py files,
and configs/preprocess.yaml.

Usage
-----
    python setup_project.py                  # scaffold under current directory
    python setup_project.py --root /my/path  # scaffold under a custom root
    python setup_project.py --dry-run        # print tree without touching disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# ── directory tree definition ─────────────────────────────────────────────────
# Each entry is either a directory path (str) or a (path, file_content) tuple.
# Use "/" as separator — Path() handles platform differences.

_DIRS: List[str] = [
    # raw inputs
    "data/raw/images",
    "data/raw/coloured_labels",
    # processed outputs
    "data/processed/images",
    "data/processed/masks",
    # patch splits
    "data/patches/train/images",
    "data/patches/train/masks",
    "data/patches/val/images",
    "data/patches/val/masks",
    "data/patches/test/images",
    "data/patches/test/masks",
    # source packages
    "src/preprocess",
    "src/datasets",
    "src/models",
    "src/analysis",
    # misc
    "configs",
    "notebooks",
    "outputs",
]

# __init__.py is written to every src/ subfolder that is a Python package.
_INIT_DIRS: List[str] = [
    "src",
    "src/preprocess",
    "src/datasets",
    "src/models",
    "src/analysis",
]

# configs/preprocess.yaml — single source of truth for all run parameters.
_CONFIG: Dict[str, Any] = {
    # Patch extraction (Node 06)
    "patch_size": 256,
    "stride": 128,
    "min_foreground": 0.05,

    # Resolution normalisation (Node 05)
    # null → auto-detect as dataset median µm/px
    "target_um_per_px": None,

    # Intensity normalisation (Node 07)
    "normalize_method": "zscore",   # zscore | clahe | minmax

    # Train/val/test split (Node 08)
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,
    "seed":        42,

    # Ignore label — unmatched pixels in label encoding (Node 03)
    "ignore_label": 255,

    # Paths (relative to project root)
    "metadata_csv":        "metadata.csv",
    "raw_image_dir":       "data/raw/images",
    "raw_label_dir":       "data/raw/coloured_labels",
    "processed_image_dir": "data/processed/images",
    "processed_mask_dir":  "data/processed/masks",
    "normed_image_dir":    "data/processed/images",
    "normed_mask_dir":     "data/processed/masks",
    "patch_dir":           "data/patches",
}


# ── scaffold logic ────────────────────────────────────────────────────────────

def scaffold(root: Path, dry_run: bool = False) -> List[Path]:
    """Create the project tree under *root*.

    Parameters
    ----------
    root:
        Absolute root directory.  Created if it does not exist.
    dry_run:
        If True, no files or directories are written.

    Returns
    -------
    list of Path
        Every path that was (or would be) created.
    """
    created: List[Path] = []

    def make_dir(rel: str) -> Path:
        p = root / rel
        if not dry_run:
            p.mkdir(parents=True, exist_ok=True)
        created.append(p)
        return p

    def write_file(rel: str, content: str) -> Path:
        p = root / rel
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text(content, encoding="utf-8")
        created.append(p)
        return p

    # Create directories.
    for d in _DIRS:
        make_dir(d)

    # Add __init__.py stubs to Python packages.
    for d in _INIT_DIRS:
        write_file(f"{d}/__init__.py", "")

    # Write configs/preprocess.yaml.
    yaml_text = _build_yaml()
    write_file("configs/preprocess.yaml", yaml_text)

    return created


def _build_yaml() -> str:
    """Serialise _CONFIG to a YAML string with section comments."""
    lines: List[str] = [
        "# MetalDAM preprocessing configuration",
        "# Edit this file to change pipeline parameters.",
        "# Do NOT hard-code values in individual scripts.",
        "",
    ]

    sections: List[Tuple[str, List[str]]] = [
        ("Patch extraction", ["patch_size", "stride", "min_foreground"]),
        ("Resolution normalisation", ["target_um_per_px"]),
        ("Intensity normalisation", ["normalize_method"]),
        ("Train / val / test split", ["train_ratio", "val_ratio", "test_ratio", "seed"]),
        ("Label encoding", ["ignore_label"]),
        ("Paths", [k for k in _CONFIG if k.endswith(("_dir", "_csv"))]),
    ]

    rendered_keys: set[str] = set()
    for title, keys in sections:
        lines.append(f"# ── {title} {'─' * max(0, 44 - len(title))}")
        for k in keys:
            if k not in _CONFIG:
                continue
            lines.append(yaml.dump({k: _CONFIG[k]}, default_flow_style=False).rstrip())
            rendered_keys.add(k)
        lines.append("")

    # Catch any keys not assigned to a section.
    remainder = [k for k in _CONFIG if k not in rendered_keys]
    if remainder:
        lines.append("# ── misc ─────────────────────────────────────────────")
        for k in remainder:
            lines.append(yaml.dump({k: _CONFIG[k]}, default_flow_style=False).rstrip())

    return "\n".join(lines) + "\n"


# ── ASCII tree printer ────────────────────────────────────────────────────────

def print_tree(root: Path, max_depth: int = 4) -> None:
    """Print an ASCII directory tree rooted at *root*."""
    print(f"\n{root.name}/")
    _tree_lines(root, prefix="", depth=0, max_depth=max_depth)
    print()


def _tree_lines(
    directory: Path,
    prefix: str,
    depth: int,
    max_depth: int,
) -> None:
    if depth >= max_depth:
        return

    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _tree_lines(entry, prefix + extension, depth + 1, max_depth)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold the MetalDAM project directory tree."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory for the project (default: current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing anything.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dry_run: bool = args.dry_run

    if dry_run:
        print(f"[dry-run] Would scaffold under: {root}")
    else:
        print(f"Scaffolding project under: {root}")

    created = scaffold(root, dry_run=dry_run)

    verb = "Would create" if dry_run else "Created"
    dirs  = sum(1 for p in created if not p.suffix)
    files = sum(1 for p in created if p.suffix)
    print(f"{verb} {dirs} directories and {files} files.\n")

    if not dry_run:
        print_tree(root)
    else:
        # In dry-run mode, render the tree from the definition, not from disk.
        print(f"\n{root.name}/  (dry-run)")
        all_paths = sorted(
            [root / d for d in _DIRS]
            + [root / d / "__init__.py" for d in _INIT_DIRS]
            + [root / "configs/preprocess.yaml"],
        )
        for p in all_paths:
            rel = p.relative_to(root)
            depth = len(rel.parts) - 1
            indent = "    " * depth
            print(f"{indent}{'└── ' if p.is_file() or p.suffix else '├── '}{p.name}")
        print()


if __name__ == "__main__":
    main()

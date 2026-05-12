# MetalDAM

A reproducible deep-learning preprocessing and segmentation pipeline for SEM (Scanning Electron Microscopy) micrographs of metallic microstructures. The pipeline converts raw SEM images and coloured phase labels into training-ready patches, trains a U-Net segmentation model, and extracts quantitative geometric features (grain diameter, phase fractions, aspect ratio, boundary density) for materials property prediction.

---

## Pipeline Overview

```
Raw SEM images + Coloured labels
         │
         ▼
  Node 01 ─ extract_scale.py     µm/pixel recovery from info bar
         │
         ▼
  Node 02 ─ crop_align.py        Remove info bar, align image to label height
         │
         ▼
  Node 03 ─ label_encoding.py    RGB label → single-channel class-index mask
         │
         ▼
  Node 04 ─ validate_pairs.py    Orphan detection + shape verification
         │
         ▼
  Node 05 ─ resolution_norm.py   Resample all pairs to a common µm/px
         │
         ▼
  Node 06 ─ patch_extract.py     Overlapping 256×256 patch tiling
         │
         ▼
  Node 07 ─ intensity_norm.py    Z-score / CLAHE / min-max normalisation
         │
         ▼
  Node 08 ─ split.py             Parent-level stratified train/val/test split
         │
         ▼
  src/train.py                   U-Net training (Dice + weighted CrossEntropy)
         │
         ▼
  src/analysis/geometric_features.py   Grain morphology feature extraction
```

---

## Project Structure

```
MetalDam/
├── environment.yml                    Conda environment specification
├── setup_project.py                   One-time scaffolding script
├── configs/
│   └── preprocess.yaml                Single source of truth for all parameters
├── data/
│   ├── raw/
│   │   ├── images/                    Original SEM .jpg (info bar intact)
│   │   └── coloured_labels/           RGB phase label .png (bar-free)
│   ├── processed/
│   │   ├── images/                    Cropped, resolution-normalised images
│   │   └── masks/                     Single-channel uint8 class-index masks
│   └── patches/
│       ├── train/{images,masks}/      Training patches
│       ├── val/{images,masks}/        Validation patches
│       └── test/{images,masks}/       Test patches
├── src/
│   ├── metadata_ledger.py             Atomic CSV interface for metadata.csv
│   ├── train.py                       U-Net training loop
│   ├── preprocess/
│   │   ├── extract_scale.py           Node 01 — µm/pixel extraction
│   │   ├── crop_align.py              Node 02 — info bar crop + alignment
│   │   ├── label_encoding.py          Node 03 — RGB → class index mask
│   │   ├── validate_pairs.py          Node 04 — orphan + shape checks
│   │   ├── resolution_norm.py         Node 05 — resample to target µm/px
│   │   ├── patch_extract.py           Node 06 — overlapping patch tiling
│   │   ├── intensity_norm.py          Node 07 — per-image intensity normalisation
│   │   └── split.py                   Node 08 — parent-level stratified split
│   ├── datasets/
│   │   ├── metaldam_dataset.py        PyTorch Dataset class + class weights
│   │   └── augmentations.py           Albumentations pipelines
│   ├── models/
│   │   └── unet.py                    4-level U-Net architecture
│   └── analysis/
│       └── geometric_features.py      Grain size, phase fractions, aspect ratio
├── metadata.csv                       Auto-generated; one row per patch
└── outputs/                           Checkpoints, logs
```

---

## Installation

**Requirements:** Anaconda or Miniconda, CUDA 12.1 compatible GPU (optional).

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate prep

# 2. Verify GPU availability (optional)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Scaffold the project directories and generate configs/preprocess.yaml
python setup_project.py
```

> **No GPU?** Edit `environment.yml` before step 1: replace `pytorch-cuda=12.1` with `cpuonly`.

After running `setup_project.py` the full `data/`, `configs/`, and `outputs/` directory tree is created and `configs/preprocess.yaml` is written with sensible defaults.

---

## Configuration

All pipeline parameters live in **`configs/preprocess.yaml`**. Never hard-code values in individual scripts — every script reads from this file at runtime.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `patch_size` | int | 256 | Square patch side length (pixels) |
| `stride` | int | 128 | Step between patches — 128 gives 50 % overlap |
| `min_foreground` | float | 0.05 | Minimum non-background fraction to keep a patch |
| `target_um_per_px` | float\|null | null | Target µm/pixel; `null` = auto (dataset median) |
| `normalize_method` | str | zscore | Intensity method: `zscore`, `clahe`, or `minmax` |
| `train_ratio` | float | 0.70 | Fraction of parent images for training |
| `val_ratio` | float | 0.15 | Fraction for validation |
| `test_ratio` | float | 0.15 | Fraction for test |
| `seed` | int | 42 | Random seed for reproducibility |
| `ignore_label` | int | 255 | Unmatched pixels in label encoding (excluded from loss) |
| `metadata_csv` | str | metadata.csv | Path to the metadata ledger |
| `raw_image_dir` | str | data/raw/images | Input SEM images |
| `raw_label_dir` | str | data/raw/coloured_labels | Input RGB labels |
| `processed_image_dir` | str | data/processed/images | Cropped images (output of Node 02) |
| `processed_mask_dir` | str | data/processed/masks | Class-index masks (output of Node 03) |
| `normed_image_dir` | str | data/processed/images | Resolution-normalised images (Node 05) |
| `normed_mask_dir` | str | data/processed/masks | Resolution-normalised masks (Node 05) |
| `patch_dir` | str | data/patches | Root directory for all patch splits |

---

## Data Preparation

Place your raw files before running the pipeline:

```
data/raw/images/           ← SEM micrographs (.jpg) with info bar still present
data/raw/coloured_labels/  ← RGB phase label images (.png), already bar-free
```

Filenames must share the same **stem** (e.g. `sample_01.jpg` ↔ `sample_01.png`). The pipeline matches pairs by stem.

### Phase colour map

| RGB | Class index | Phase |
|-----|:-----------:|-------|
| (255, 0, 255) | 0 | Background / Defect |
| (0, 255, 0) | 1 | Austenite |
| (128, 0, 255) | 2 | Matrix |
| (255, 255, 0) | 3 | Martensite-Austenite (MA) |
| (255, 0, 0) | 4 | Precipitate |

---

## Step-by-Step Pipeline Execution

Run all commands from the **project root** with the `prep` environment active.

### Node 01 — Extract µm/pixel scale

Reads the white scale bar in the SEM info strip (bottom of the image) and records µm/pixel for every image. This **must run before any cropping**.

```bash
python src/preprocess/extract_scale.py --config configs/preprocess.yaml
```

- **Input:** `data/raw/images/*.jpg`
- **Output:** prints µm/pixel per image; use `--image` for a single file
- **Optional:** pass `--image path/to/image.jpg` to process one file

```bash
# Single image
python src/preprocess/extract_scale.py --config configs/preprocess.yaml \
    --image data/raw/images/sample_01.jpg
```

---

### Node 02 — Crop info bar and align to label height

Removes the SEM info bar from each image by cropping to exactly `label.shape[0]` rows. Saves cropped images to `data/processed/images/`.

```bash
python src/preprocess/crop_align.py --config configs/preprocess.yaml
```

- **Input:** `data/raw/images/` + `data/raw/coloured_labels/`
- **Output:** `data/processed/images/` (cropped, bar-free SEM images)
- **Raises:** `ValueError` with details if any image/label shape does not match after crop

---

### Node 03 — RGB label → class-index mask

Converts coloured RGB label images into single-channel uint8 masks where each pixel value is a class index (0–4) or 255 (ignore). Tolerance L2 < 20 handles JPEG colour bleeding.

```bash
python src/preprocess/label_encoding.py --config configs/preprocess.yaml
```

Or validate a single label interactively:

```bash
python src/preprocess/label_encoding.py data/raw/coloured_labels/sample_01.png
```

- **Input:** `data/raw/coloured_labels/*.png`
- **Output:** `data/processed/masks/*.png` (single-channel uint8)
- **Prints:** per-class pixel counts and warns if any class covers < 0.5 %

---

### Node 04 — Validate image/label pairs

Checks that every image has a matching label (same stem), that no orphans exist, and that all matched pairs have identical H×W dimensions.

```bash
python src/preprocess/validate_pairs.py --config configs/preprocess.yaml
```

- **Input:** `data/processed/images/` + `data/raw/coloured_labels/`
- **Output:** prints a summary table; raises `RuntimeError` listing **all** failures if any pair is invalid
- **Fix:** resolve every reported issue before proceeding to Node 05

---

### Node 05 — Normalise resolution

Resamples all image/mask pairs to a common physical resolution (`target_um_per_px`). When `target_um_per_px: null` in the config, the dataset median is used automatically.

```bash
python src/preprocess/resolution_norm.py --config configs/preprocess.yaml
```

- **Input:** `data/processed/images/` + `data/processed/masks/` + `metadata.csv`
- **Output:** overwrites `data/processed/images/` and `data/processed/masks/`; updates `um_per_pixel`, `height`, `width` in `metadata.csv`
- **Note:** masks are always resized with `INTER_NEAREST` — never `INTER_LINEAR` or `INTER_CUBIC`

---

### Node 06 — Extract overlapping patches

Slides a 256×256 window (stride 128) over every image/mask pair and saves patches. Drops patches where foreground pixels (class ≠ 0) are below `min_foreground`.

```bash
python src/preprocess/patch_extract.py --config configs/preprocess.yaml
```

- **Input:** `data/processed/images/` + `data/processed/masks/`
- **Output:** `data/patches/train/images/`, `data/patches/train/masks/` (split assignment happens in Node 08; patches land in `train/` by default until then)
- **Appends** patch rows to `metadata.csv` (filename, parent_id, patch_y, patch_x, class_histogram, …)

Process a single split only:

```bash
python src/preprocess/patch_extract.py --config configs/preprocess.yaml --split train
```

---

### Node 07 — Intensity normalisation

Normalises each patch image using the method set in `normalize_method`. For `zscore`, computes dataset-wide mean/std from training patches first.

```bash
python src/preprocess/intensity_norm.py --config configs/preprocess.yaml
```

Print dataset statistics only (no files written):

```bash
python src/preprocess/intensity_norm.py --config configs/preprocess.yaml --stats-only
```

- **Input:** `data/patches/{train,val,test}/images/`
- **Output:** `data/patches/{train,val,test}/images_norm/` (float32 [0,1] stored as uint8 PNG)
- **Methods:**
  - `zscore` — zero-mean / unit-std, clipped ±3σ, rescaled to [0,1]
  - `clahe` — contrast-limited adaptive histogram equalisation
  - `minmax` — simple min-max stretch

---

### Node 08 — Parent-level stratified split

Assigns every parent image to train / val / test and propagates the label to all its child patches. Stratifies by `dominant_class` (most frequent class per parent).

```bash
python src/preprocess/split.py --config configs/preprocess.yaml
```

- **Input:** `metadata.csv`
- **Output:** `metadata.csv` with `split` column filled in; prints class distribution per split
- **Critical:** splitting always happens at the **parent image** level — all patches from the same parent land in the same split to prevent data leakage

---

## Training

After completing all 8 pipeline nodes:

```bash
python src/train.py --config configs/preprocess.yaml --epochs 50 --lr 1e-4
```

Optional overrides:

```bash
python src/train.py --config configs/preprocess.yaml \
    --epochs 100 \
    --lr 3e-4 \
    --batch-size 16
```

- **Model:** 4-level U-Net, ~31 M parameters (base_features=64)
- **Loss:** 0.5 × Dice + 0.5 × weighted CrossEntropyLoss (`ignore_index=255`)
- **Optimiser:** AdamW (`weight_decay=1e-4`)
- **Scheduler:** CosineAnnealingLR (`eta_min = lr/100`)
- **Checkpoint:** `outputs/best_model.pth` saved on each validation IoU improvement

---

## Geometric Feature Extraction

After inference, extract quantitative grain morphology features from predicted masks:

```bash
python src/analysis/geometric_features.py
```

Programmatic usage:

```python
import cv2
import numpy as np
from src.analysis.geometric_features import extract_geometric_features, features_to_vector

mask = cv2.imread("outputs/pred_mask.png", cv2.IMREAD_GRAYSCALE)
um_per_pixel = 0.012   # read from metadata.csv

features = extract_geometric_features(mask, um_per_pixel)
print(features["mean_grain_diameter_um"])   # µm
print(features["phase_fractions"])          # {0: 0.19, 1: 0.33, ...}
print(features["boundary_density_um"])      # µm⁻¹

vec = features_to_vector(features)          # (11,) float64 for regression
```

Outputs per image:

| Feature | Unit | Description |
|---------|------|-------------|
| `mean_grain_diameter_um` | µm | Mean equivalent circular grain diameter |
| `phase_fractions` | — | Fraction of valid pixels per class |
| `mean_aspect_ratio` | — | Mean of major / minor axis length |
| `boundary_density_um` | µm⁻¹ | Total boundary length / total imaged area |

---

## File Reference

### `environment.yml`
Conda environment definition. Channels: `pytorch`, `nvidia`, `conda-forge`. Key packages: Python 3.11, PyTorch 2.x + CUDA 12.1, OpenCV, scikit-image, albumentations, pytesseract.

---

### `setup_project.py`
One-time scaffolding script. Creates the full `data/`, `src/`, `configs/`, `notebooks/`, `outputs/` directory tree, adds `__init__.py` to each `src/` package, and writes `configs/preprocess.yaml`.

| Function | Description |
|----------|-------------|
| `scaffold(root, dry_run)` | Creates directories and files; returns list of created paths |
| `print_tree(root, max_depth)` | Prints ASCII directory tree |

```bash
python setup_project.py --dry-run          # preview without writing
python setup_project.py                    # scaffold under current directory
python setup_project.py --root /path/to/project
```

---

### `src/metadata_ledger.py`
Append-only interface to `metadata.csv`. Enforces column schema, validates required fields, and performs atomic writes (temp file → rename) so a crash never leaves a corrupt CSV.

| Class / Method | Description |
|----------------|-------------|
| `MetadataLedger(csv_path)` | Load or create `metadata.csv` |
| `.add_entry(**kwargs)` | Append one row; raises `KeyError` on missing required fields |
| `.add_entries(rows)` | Batch append (single concat — O(1) overhead) |
| `.get_split(name)` | Return filtered DataFrame for `"train"`, `"val"`, or `"test"` |
| `.update_split_assignments(dict)` | Bulk-set `split` column from `{parent_id: split}` mapping |
| `.save()` | Atomic write: tmp file → `os.replace` |
| `.summary()` | Print counts, mean µm/px, and dataset-wide class distribution |

Required columns: `filename`, `parent_id`, `um_per_pixel`, `height`, `width`, `patch_y`, `patch_x`, `class_histogram`.

---

### `src/train.py`
End-to-end training script for U-Net segmentation.

| Class / Function | Description |
|-----------------|-------------|
| `DiceLoss` | Soft multi-class Dice loss; ignores label 255 |
| `CombinedLoss` | 0.5 × Dice + 0.5 × weighted CrossEntropy |
| `mean_iou(logits, targets)` | Macro-mean IoU, ignoring label 255 |
| `run_epoch(model, loader, ...)` | One training or validation epoch; returns (loss, iou) |
| `train(cfg)` | Full training loop with checkpointing |

---

### `src/preprocess/extract_scale.py` — Node 01

Recovers µm/pixel using a priority cascade:
1. CSV lookup (`um_per_pixel` column in `metadata.csv`)
2. OCR of info bar via `pytesseract` (parses `"1 µm"`, `"500 nm"`, etc.)
3. Connected-component detection of the white scale bar

| Function | Description |
|----------|-------------|
| `extract_um_per_pixel(img_path, csv_path, known_length_um, use_ocr)` | Returns µm/pixel as float |

Key constants: `_INFO_BAR_FRACTION = 0.12`, `_SCALE_REGION_FRACTION = 0.35`, `_BAR_THRESHOLD = 200`.

---

### `src/preprocess/crop_align.py` — Node 02

Removes the SEM info bar by cropping to `label.shape[0]` rows. Auto-detects the bar edge via row-wise pixel variance (for logging), but the actual crop is always governed by the label height.

| Function | Description |
|----------|-------------|
| `crop_to_label_height(img_path, label_path, out_dir)` | Crop, assert shapes match, save; returns crop row index |

Raises `ValueError` with full details (both paths and both shapes) on mismatch.

---

### `src/preprocess/label_encoding.py` — Node 03

Converts RGB label images to single-channel class-index masks. Assigns each pixel to the nearest colour in the colour map by L2 distance; unmatched pixels (distance > tolerance) become 255.

| Function | Description |
|----------|-------------|
| `rgb_to_class_mask(label_rgb, color_map, tolerance=20.0)` | Returns H×W uint8 class mask |
| `validate_color_map(label_rgb, color_map, ...)` | Prints per-class counts; warns if any class < 0.5 % |
| `METALDAM_COLOR_MAP` | Canonical `{(R,G,B): class_index}` dict |

Algorithm: vectorised L2 via numpy broadcasting — no Python loops over pixels.

---

### `src/preprocess/validate_pairs.py` — Node 04

Validates image/label pairs by stem matching and H×W comparison. Collects **all** failures before raising so every issue can be fixed in one pass.

| Function | Description |
|----------|-------------|
| `validate_pairs(images_dir, labels_dir)` | Returns DataFrame; raises `RuntimeError` listing all failures |

DataFrame columns: `filename`, `has_image`, `has_label`, `shapes_match`, `notes`.

---

### `src/preprocess/resolution_norm.py` — Node 05

Resamples image/mask pairs to a common µm/pixel. Masks always use `INTER_NEAREST`; images use `INTER_AREA` (downscale) or `INTER_LINEAR` (upscale).

| Function | Description |
|----------|-------------|
| `normalize_resolution(img, mask, src_um_per_px, tgt_um_per_px)` | Returns (resized_img, resized_mask, new_um_per_px) |
| `batch_normalize(metadata_csv, ...)` | Processes all pairs listed in CSV; updates `um_per_pixel`, `height`, `width` |

When `tgt_um_per_px=None`, uses the dataset median (recommended).

---

### `src/preprocess/patch_extract.py` — Node 06

Slides a `patch_size × patch_size` window with the given stride. Drops patches where the foreground fraction (class ≠ 0) falls below `min_foreground`.

| Function | Description |
|----------|-------------|
| `extract_patches(img, mask, patch_size, stride, min_foreground, parent_file, um_per_pixel)` | Returns list of patch dicts |
| `save_patches(patches, out_dir)` | Saves to `out_dir/images/` and `out_dir/masks/`; returns metadata records |

Patch filenames: `{parent}__y{y:05d}_x{x:05d}.png`.

---

### `src/preprocess/intensity_norm.py` — Node 07

Normalises single-channel SEM patches to float32 [0, 1]. Supports per-image and dataset-wide z-score using Welford's online algorithm.

| Function | Description |
|----------|-------------|
| `normalize_intensity(img, method, global_mean, global_std)` | Returns H×W float32 in [0, 1] |
| `compute_dataset_stats(img_paths)` | Returns global (mean, std) across all images; single-pass, memory-efficient |

Methods: `zscore` (default), `clahe`, `minmax`.

---

### `src/preprocess/split.py` — Node 08

Stratified train/val/test split at the **parent image** level. All patches sharing a `parent_id` receive the same split to prevent spatial leakage.

| Function | Description |
|----------|-------------|
| `split_dataset(metadata_csv, train, val, test, seed, stratify_col)` | Adds `split` column; saves CSV; prints distribution |

Uses two-stage `sklearn.train_test_split`. Falls back to unstratified split if any stratum has fewer than 2 samples.

---

### `src/datasets/metaldam_dataset.py`
PyTorch `Dataset` for patch-based training.

| Class / Method | Description |
|----------------|-------------|
| `MetalDAMDataset(metadata_csv, split, image_dir, mask_dir, transform)` | Filters CSV to the given split |
| `__getitem__(idx)` | Returns `(image [1,H,W] float32, mask [H,W] int64)` |
| `get_class_weights()` | Inverse-frequency weights as `torch.Tensor (N_CLASSES,)` for `CrossEntropyLoss` |

---

### `src/datasets/augmentations.py`
Albumentations pipelines. Spatial transforms are applied identically to both image and mask via `additional_targets={"mask": "mask"}`.

| Function | Description |
|----------|-------------|
| `get_train_transforms(p_flip=0.5)` | HorizontalFlip, VerticalFlip, RandomRotate90, GaussianBlur, GaussNoise, RandomBrightnessContrast |
| `get_val_transforms()` | Identity (no transforms) |
| `get_tensor_transforms()` | ToTensorV2 for use after augmentation |

Banned transforms: `ElasticTransform` (distorts grain morphology), `ShiftScaleRotate` with scale ≠ 1 (breaks µm/px calibration), heavy colour jitter (SEM is grayscale).

---

### `src/models/unet.py`
4-level U-Net for single-channel SEM segmentation.

- **Input:** `[B, 1, H, W]` float32
- **Output:** `[B, N_CLASSES, H, W]` logits
- **Encoder:** 4 stages of `(Conv-BN-ReLU) × 2` + MaxPool; feature widths 64, 128, 256, 512
- **Bottleneck:** 1024 features
- **Decoder:** ConvTranspose2d upscale + skip concatenation at each level
- **Head:** 1×1 Conv to `n_classes`

| Class | Description |
|-------|-------------|
| `UNet(in_channels=1, n_classes=5, base_features=64)` | Full U-Net |
| `_Block(in_ch, out_ch)` | Two Conv-BN-ReLU layers |

---

### `src/analysis/geometric_features.py`
Extracts quantitative grain morphology features using `skimage.measure.regionprops`. All pixel measurements are multiplied by `um_per_pixel` before returning.

| Function | Description |
|----------|-------------|
| `extract_geometric_features(mask, um_per_pixel)` | Returns dict of grain features |
| `features_to_vector(features)` | Flattens to `(11,)` float64 array for regression |

Feature vector layout: `[mean_diameter, phase_0, …, phase_4, mean_aspect_ratio, boundary_density]`.

---

## metadata.csv Schema

Every preprocessing node appends to this file. **Never delete columns.**

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Patch filename |
| `parent_id` | str | Stem of the source SEM image |
| `split` | str | `train` / `val` / `test` (set by Node 08) |
| `um_per_pixel` | float | µm per pixel after resolution normalisation |
| `magnification_kx` | float | From SEM info bar (nullable) |
| `kV` | float | Accelerating voltage (nullable) |
| `height` | int | Patch height (px) |
| `width` | int | Patch width (px) |
| `patch_y` | int | Top-left y in parent image |
| `patch_x` | int | Top-left x in parent image |
| `class_histogram` | str | JSON `{"0": n, "1": n, …}` |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AssertionError: shape mismatch` | Cropped image ≠ label size | Use `label.shape[0]` as crop height, not a hard-coded row |
| Grain size values implausibly large | `um_per_pixel` not applied to regionprops | Multiply every measurement by `um_per_pixel` |
| Validation Dice inflated | Split done at patch level | Re-split at parent image level with `split.py` |
| Class `255` in loss | Ignore label leaking into training | Set `ignore_index=255` in `CrossEntropyLoss` |
| Blurry mask boundaries | Wrong interpolation on mask resize | Use `cv2.INTER_NEAREST` for all mask resizes |

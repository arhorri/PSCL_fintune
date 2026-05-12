# MetalDAM Preprocessing Pipeline

Deep-learning preprocessing pipeline for SEM micrographs: scale calibration → label encoding → patch extraction → U-Net / GAN training.

---

## Project Structure

```
metaldam_project/
├── CLAUDE.md
├── configs/
│   └── preprocess.yaml          # single source of truth for all run params
├── data/
│   ├── raw/
│   │   ├── images/              # original SEM .jpg with info bar intact
│   │   └── coloured_labels/     # original RGB label .png
│   ├── processed/
│   │   ├── images/              # cropped, bar-free, resolution-normalised
│   │   └── masks/               # single-channel uint8 class index maps
│   └── patches/
│       ├── train/{images,masks}/
│       ├── val/{images,masks}/
│       └── test/{images,masks}/
├── src/
│   ├── preprocess/
│   │   ├── extract_scale.py     # Node 01 — µm/pixel recovery
│   │   ├── crop_align.py        # Node 02 — info bar crop + pair alignment
│   │   ├── label_encoding.py    # Node 03 — RGB → class index
│   │   ├── validate_pairs.py    # Node 04 — orphan + shape checks
│   │   ├── resolution_norm.py   # Node 05 — rescale to target µm/px
│   │   ├── patch_extract.py     # Node 06 — overlapping patch tiling
│   │   ├── intensity_norm.py    # Node 07 — per-image z-score / CLAHE
│   │   └── split.py             # Node 08 — parent-level stratified split
│   ├── datasets/
│   │   └── metaldam_dataset.py  # PyTorch Dataset + class weights
│   ├── models/
│   │   ├── unet.py
│   │   └── gan.py
│   └── analysis/
│       └── geometric_features.py  # grain size, phase fraction, aspect ratio
├── metadata.csv                 # one row per PARENT image; patches inherit
└── outputs/
```

---

## Environment

```bash
# install
pip install opencv-python numpy pandas scikit-learn albumentations \
            torch torchvision scikit-image pytesseract pyyaml tqdm

# verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# run full pipeline (reads configs/preprocess.yaml)
python src/preprocess/run_pipeline.py

# run a single step
python src/preprocess/extract_scale.py --config configs/preprocess.yaml

# quick smoke test on 5 images
python src/preprocess/run_pipeline.py --smoke-test
```

---

## Key Parameters (`configs/preprocess.yaml`)

```yaml
patch_size: 256           # px — input size expected by U-Net
stride: 128               # 50 % overlap
target_um_per_px: null    # null = auto (median of dataset); or set float e.g. 0.012
normalize_method: zscore  # zscore | clahe | minmax
min_foreground: 0.05      # drop patches with < 5 % non-background pixels
train_ratio: 0.70
val_ratio:   0.15
test_ratio:  0.15
seed: 42
ignore_label: 255         # unmatched pixels in label encoding
```

Change parameters only in this file. Scripts read from it at runtime.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AssertionError: shape mismatch` | Cropped image ≠ label size | Use `label.shape[0]`, not a hard-coded row |
| Grain size values implausibly large | `um_per_pixel` not applied | Multiply regionprops output by `um_per_pixel` |
| Validation Dice inflated | Split done at patch level | Re-split at parent image level |
| `255` class in loss | Ignore label leaking | Set `ignore_index=255` in CrossEntropyLoss |
| Blurry boundaries in masks | Wrong interpolation on resize | Use `INTER_NEAREST` for all mask resizes |

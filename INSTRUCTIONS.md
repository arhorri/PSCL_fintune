# MetalDAM → PSCL Finetuning: Instructions

## Overview

This project adapts the MetalDAM preprocessed patch dataset to finetune PSCL
(Patch-Sampled Contrastive Learning), a self-supervised segmentation model originally
built for aluminum microstructure images.

**What this project does:**
- Bridges the format gap between MetalDAM patches and PSCL's expected inputs
- Runs PSCL's self-supervised contrastive pretraining on MetalDAM data (Stage 1)
- Finetunes the pretrained encoder for steel microstructure segmentation (Stage 2)

**What this project does NOT touch:**
The entire `PSCL/PSCL/` source directory is read-only. All new code lives here in
`Pscl_fintune/`. PSCL model classes are imported via `sys.path` at runtime.

---

## Prerequisites

### 1. Conda environment
```bash
conda activate prep   # MetalDAM environment — has PyTorch, OpenCV, torchvision
```

### 2. GPU
PSCL requires CUDA. Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If this prints `False`, set `GPU = '0'` in `train_metaldam.py` to the correct device index.

### 3. Data paths
MetalDAM must have been fully preprocessed (all 8 nodes run). Verify:
```
MetalDam/data/patches/
    train/images_norm/   <- 788 .png files
    train/masks/         <- 788 .png files
    val/images_norm/     <- 168 .png files
    val/masks/           <- 168 .png files
    test/images_norm/    <- 192 .png files
    test/masks/          <- 192 .png files
```
The path is hard-coded in `train_metaldam.py` as:
```
METALDAM_PATCHES = '/mnt/e/Afile/1-Python/MetalDam/data/patches'
```
Change it there if your data is elsewhere.

---

## File-by-File Explanation

### `data_metaldam.py` — Dataset adapter

**Purpose:** Converts MetalDAM patches to the exact tensor format PSCL expects.

PSCL was built for 376×376 RGB images with 4 classes encoded as one-hot tensors.
MetalDAM produces 256×256 grayscale images with 5 classes as integer indices.
This file bridges that gap on-the-fly so the patches on disk are never modified.

**Three classes inside:**

| Class | Mirrors | Used by |
|-------|---------|---------|
| `MetalDAMDataset` | `Data_preheat` | Supervised finetuning (train/val/test) |
| `MetalDAMMoCoDataset` | `MoCoData_preheat` | Self-supervised pretraining — unlabeled pool |
| `MetalDAMMoCoDatasetSup` | `MoCoData_preheat_sup` | Self-supervised pretraining — labeled guidance |

**What every `__getitem__` does (in order):**

1. Load image as grayscale uint8 PNG
2. Resize 256×256 → 376×376 with `INTER_LINEAR`
3. Repeat single channel three times → (H, W, 3) RGB array
4. Apply ImageNet normalisation (`mean=(0.491, 0.482, 0.447)`, `std=(0.247, 0.243, 0.262)`)
5. Apply random flip + random rot90 spatial augmentation (training only)
6. Load mask as uint8 PNG
7. Remap class indices before resize:

   | MetalDAM | Phase | PSCL |
   |----------|-------|------|
   | 0 | Background | **255 (ignore)** |
   | 1 | Austenite | 0 |
   | 2 | Matrix | 1 |
   | 3 | Martensite-Austenite | 2 |
   | 4 | Precipitate | 3 |

8. Resize mask with `INTER_NEAREST` (mandatory — any other mode creates phantom classes)
9. Convert to one-hot tensor (4, H, W) float32; ignore pixels (255) → all-zero channels

`MetalDAMMoCoDataset` generates two differently-augmented views of each image for
contrastive learning (query and key), returned as a single (6, H, W) tensor.

`MetalDAMMoCoDatasetSup` does the same but also returns the one-hot mask scaled ×255,
concatenated as a (10, H, W) tensor — this is the labeled guidance sample that PSCL
uses to decide which patches are semantically similar during pretraining.

---

### `finetune.py` — Training functions

**Purpose:** Contains the two training loops and the checkpoint loader.

**`load_moco(base_encoder, checkpoint_path)`**
Loads a PSCL MoCo checkpoint into a plain UNet for finetuning. Strips the
`encoder_q.` prefix from the state dict keys and applies them with `strict=False`
(the projection head keys are silently dropped — only the UNet backbone is kept).

**`SelfSupervised_MetalDAM(cfg)`**
PSCL self-supervised pretraining loop. Matches the structure of PSCL's original
`SelfSupervised()` function but uses MetalDAM dataset classes.

- Unlabeled pool: `MetalDAMMoCoDataset` (train split, 788 patches)
- Labeled guidance: `MetalDAMMoCoDatasetSup` (val split, 168 patches)
- At each batch: one supervised sample is prepended to the unsupervised batch,
  the MoCo model uses its label to identify semantically similar patches for
  the dense contrastive loss
- Loss: global InfoNCE loss + multi-scale dense patch contrastive loss, weighted
  by `moco_denseloss_ratio`
- Saves checkpoint every `self_save_epoch` epochs as `moco{epoch}.pt`

**`Finetune_MetalDAM(cfg)`**
Supervised segmentation finetuning. Matches PSCL's `Finetune()` but:
- Uses all three MetalDAM splits (train / val / test)
- Validates on val split every `test_freq` epochs and saves best checkpoint (`fine.pt`)
- Prints final mIoU + per-class IoU on test split at the end

Loss: combined `(1 - dice_bce_ratio) × BCE + dice_bce_ratio × Dice`
Optimizer: Adam with two learning rate groups — encoder layers get `fineturn_lr_en = 1e-4`,
decoder and upsampling layers get `fineturn_lr_de = 1e-3`.

---

### `config_metaldam.py` — Configuration

**Purpose:** Holds all hyperparameters in one place and wires them to the training functions.

**`MetalDAMConfig`**
A plain Python class (no inheritance from PSCL) that stores every hyperparameter the
training functions need. The defaults are tuned for MetalDAM:

| Key parameter | Default | Why |
|---------------|---------|-----|
| `weight` | `[1, 1, 5, 5]` | Upweight MA and Precipitate (rare phases) |
| `dice_weight` | `[1, 1, 1, 1]` | Equal Dice contribution across classes |
| `moco_denseloss_ratio` | `0.7` | Emphasise local patch loss over global |
| `temperature` | `0.07` | Standard MoCo temperature |
| `self_max_epoch` | `200` | Matches PSCL paper pretraining length |
| `fine_max_epoch` | `50` | Full-annotation supervised finetuning |
| `fineturn_lr_en` | `1e-4` | Preserve pretrained encoder features |
| `fineturn_lr_de` | `1e-3` | Train fresh decoder from scratch |

The `tmp` attribute sets the output directory name, mirroring PSCL's convention:
`{method}_{backbone}_{tag}/` → e.g., `self_UNet_metaldam/`

**`run(method, tt, data_dir, ...)`**
One-line entry point that builds the config, seeds all random number generators, and
calls the appropriate training function. Used by `train_metaldam.py`.

---

### `train_metaldam.py` — Entry point

**Purpose:** CLI script that runs Stage 1 (pretraining) and/or Stage 2 (finetuning).

Adds `PSCL/PSCL/` to `sys.path` before importing anything, so PSCL's model classes
(`UNet`, `MoCo_DenseModel`, `utils.py`) are available without modifying PSCL.
Changes the working directory to `Pscl_fintune/` so all output paths are written here.

Three constants to edit before running:

```python
METALDAM_PATCHES = '/mnt/e/Afile/1-Python/MetalDam/data/patches'  # path to data
PRETRAIN_EPOCHS  = 200    # how many pretraining epochs to run
GPU              = '0'    # which GPU (CUDA_VISIBLE_DEVICES)
```

---

## Step-by-Step Run Guide

### Step 1 — Verify the environment

```bash
conda activate prep
python -c "import torch, cv2, torchvision; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```

Expected output: `PyTorch 2.x.x | CUDA True`

### Step 2 — Verify MetalDAM data exists

```bash
ls /mnt/e/Afile/1-Python/MetalDam/data/patches/train/images_norm/ | wc -l   # should print 788
ls /mnt/e/Afile/1-Python/MetalDam/data/patches/val/images_norm/   | wc -l   # should print 168
ls /mnt/e/Afile/1-Python/MetalDam/data/patches/test/images_norm/  | wc -l   # should print 192
```

If the counts are wrong, re-run the MetalDAM pipeline first:
```bash
cd /mnt/e/Afile/1-Python/MetalDam
python src/preprocess/run_pipeline.py
```

### Step 3 — Check GPU index

Open `train_metaldam.py` and confirm `GPU = '0'` matches your available GPU:
```bash
nvidia-smi   # shows available GPU indices
```

### Step 4 — Run Stage 1: Self-supervised pretraining

```bash
cd /mnt/e/Afile/1-Python/Pscl_fintune
conda activate prep
python train_metaldam.py --stage self
```

This runs for `PRETRAIN_EPOCHS = 200` epochs. A checkpoint is saved every 10 epochs.
Progress is printed to the terminal and logged to:
```
self_UNet_metaldam_Numf/f/log_self.txt
```

To check the most recent checkpoint:
```bash
ls -lh self_UNet_metaldam_Numf/f/moco*.pt
```

### Step 5 — Run Stage 2: Supervised finetuning

```bash
python train_metaldam.py --stage fine
```

This loads `moco200.pt` from Stage 1 and finetunes for 50 epochs.
The best checkpoint (based on val accuracy) is saved as:
```
fine_UNet_metaldam_Numf/f/fine.pt
```

Progress and metrics are logged to:
```
fine_UNet_metaldam_Numf/f/log_fine.txt
```

### Step 6 — Run both stages in sequence (recommended)

```bash
python train_metaldam.py
```

Equivalent to running `--stage self` then `--stage fine` automatically.

### Step 7 — Read the results

At the end of Stage 2, the terminal prints the final test-split metrics:

```
[test] ACC 0.xxxx  mIoU 0.xxxx  IoU1 0.xxxx  IoU2 0.xxxx  IoU3 0.xxxx
```

| Metric | Meaning |
|--------|---------|
| ACC | Pixel accuracy across all 4 classes |
| mIoU | Mean IoU across Austenite, Matrix, MA |
| IoU1 | Austenite (PSCL class 0) |
| IoU2 | Matrix (PSCL class 1) |
| IoU3 | Martensite-Austenite (PSCL class 2) |

Predicted segmentation maps are saved as greyscale PNGs alongside the log:
```
fine_UNet_metaldam_Numf/f/epoch_{N}_test/
```

---

## Output Directory Layout

After a complete run, `Pscl_fintune/` will contain:

```
Pscl_fintune/
├── self_UNet_metaldam_Numf/
│   └── f/
│       ├── log_self.txt          <- training loss per epoch
│       ├── moco0.pt              <- checkpoint at epoch 0
│       ├── moco10.pt             <- checkpoint at epoch 10
│       ├── ...
│       └── moco200.pt            <- final pretrain checkpoint (loaded by Stage 2)
│
└── fine_UNet_metaldam_Numf/
    └── f/
        ├── log_fine.txt          <- train loss + val metrics per epoch
        ├── fine.pt               <- best finetuned model (save on val acc improvement)
        └── epoch_{N}_test/       <- greyscale prediction images on test split
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: .../images_norm/` | MetalDAM patches missing | Re-run MetalDAM pipeline (Step 2) |
| `CUDA out of memory` | Batch size too large for GPU | Reduce `batch_size` or `self_batch_size` in `config_metaldam.py` |
| `No such file: moco200.pt` when running `--stage fine` | Stage 1 not finished | Run `--stage self` first, or change `FINETUNE_CKPT_EP` to an existing epoch |
| `ModuleNotFoundError: No module named 'model'` | PSCL path wrong | Check `_PSCL_SRC` in `train_metaldam.py` points to the correct directory |
| `RuntimeError: Expected all tensors to be on the same device` | No GPU / wrong device | Set `GPU = '0'` and verify CUDA is available |
| `IndexError: list index out of range` in `networks.py` line 244 | `HeadNrom` list too short | Ensure `HeadNrom=['LN', '']` (two elements) in `config_metaldam.py` |
| `IndexError: index 6 is out of bounds for axis 0 with size 6` in `model.py` | PSCL hardcodes 6 production groups; MetalDAM has 29+ parent images | Fixed in `data_metaldam.py`: IDs bucketed into 4 groups via `parent_num % 4` |
| Low mIoU on Precipitate (class 3) | Rare class underrepresented | Increase `weight[3]` in `config_metaldam.py` (default `[1,1,5,5]`) |

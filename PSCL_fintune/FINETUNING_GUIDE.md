# MetalDAM → PSCL Finetuning Guide

This document describes how to use the MetalDAM preprocessed patch dataset to finetune the PSCL segmentation model step by step.

---

## Format Mismatch Overview

Before any training can happen, four differences between what MetalDAM produces and what PSCL expects must be bridged:

| Property | MetalDAM output | PSCL expects | Bridge |
|----------|----------------|--------------|--------|
| Image size | 256×256 px | 376×376 px | resize |
| Channels | 1 (grayscale) | 3 (RGB) | repeat channel ×3 |
| Normalisation | z-score → `[0, 1]` | ImageNet mean/std | remap |
| Classes | 5 (indices 0–4) + 255 ignore | 4 (one-hot encoded) | Background(0) → ignore; shift 1–4 → 0–3 |

---

## Flowchart

```
╔══════════════════════════════════════════════════════════════════╗
║           MetalDAM Pipeline  (already complete)                  ║
║  42 raw SEM images  →  8 preprocessing nodes  →  1,148 patches   ║
║  data/patches/{train(788), val(168), test(192)}/                  ║
║       images_norm/  <--  uint8 PNG  (256x256, grayscale)         ║
║       masks/        <--  uint8 PNG  (256x256, class indices 0-4) ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               |
                               v
╔══════════════════════════════════════════════════════════════════╗
║                    DATA ADAPTATION LAYER                         ║
║          (new file: Pscl_finetune/data_metaldam.py)              ║
║                                                                  ║
║  1. Load image as uint8 PNG  ->  float32 / 255                   ║
║  2. Grayscale -> 3-channel:  np.repeat(img, 3, axis=0)           ║
║  3. Resize 256x256 -> 376x376  (INTER_LINEAR for image)          ║
║  4. Normalise with ImageNet mean/std                              ║
║  5. Class remap:                                                  ║
║       0 Background  -> 255 (ignore)                              ║
║       1 Austenite   -> 0                                         ║
║       2 Matrix      -> 1                                         ║
║       3 MA          -> 2                                         ║
║       4 Precipitate -> 3                                         ║
║  6. Mask resize -> INTER_NEAREST (class indices must not blend)  ║
║  7. Mask -> one-hot:  get_one_hot(mask, num_classes=4)           ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               |
                               v
╔══════════════════════════════════════════════════════════════════╗
║              PSCL CONFIG CHANGES  (minimal)                      ║
║  config.py  ->  data_dir pointing at MetalDAM patches            ║
║  config.py  ->  tune finetune_lr_en / finetune_lr_de             ║
║  config_run.py  ->  update class weight vectors (still length 4) ║
║  No architecture changes needed — 4 classes, same UNet depth     ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               |
                +--------------+--------------+
                |                             |
                v                             v
  ╔═════════════════════════╗   ╔═════════════════════════════════╗
  ║  PATH A -- PSCL         ║   ║  PATH B -- direct finetuning    ║
  ║  self-supervised        ║   ║  (skip pretraining)             ║
  ║  pretraining first      ║   ║                                 ║
  ║                         ║   ║  Start from ImageNet-init or    ║
  ║  MetalDAM train patches ║   ║  random encoder weights         ║
  ║  used as unlabeled pool ║   ║                                 ║
  ║  + small labeled sample ║   ║  Run as ablation baseline to    ║
  ║  -> MoCo / SimCLR loss  ║   ║  measure PSCL's actual benefit  ║
  ╚══════════════╦══════════╝   ╚══════════════╦══════════════════╝
                 |                             |
                 +--------------+--------------+
                                |
                                v
╔══════════════════════════════════════════════════════════════════╗
║                         FINETUNING                               ║
║  config_run.py  ->  Finetune()                                   ║
║                                                                  ║
║  Load:    pretrained PSCL encoder weights (UNet_encode)          ║
║  Attach:  fresh UNet_decode (4-class segmentation head)          ║
║  Freeze:  encoder for first N epochs (optional warmup)           ║
║  Loss:    Dice + BCE with class weights for rare phases          ║
║  Ignore:  remapped label 255 excluded from loss                  ║
║                                                                  ║
║  Train split:  788 patches                                       ║
║  Val split:    168 patches  <-- save best checkpoint on mIoU     ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               |
                               v
╔══════════════════════════════════════════════════════════════════╗
║                        EVALUATION                                ║
║  Test split:  192 patches                                        ║
║  Metrics:  mIoU, per-class IoU (Austenite / Matrix / MA /        ║
║            Precipitate), Dice coefficient, pixel accuracy        ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Step-by-Step Explanation

### Step 1 — MetalDAM Pipeline (prerequisite, already done)

The MetalDAM pipeline has already processed 42 raw SEM micrographs through 8 nodes and produced **1,148 labelled patches** at `MetalDam/data/patches/`. Each patch is:

- **Image:** 256×256 grayscale uint8 PNG, z-score normalised, stored in `images_norm/`
- **Mask:** 256×256 uint8 PNG with class indices 0–4, stored in `masks/`

The split is fixed at the **parent-image level** — all patches from the same micrograph land in the same split — preventing spatial data leakage. Distribution: train 788, val 168, test 192.

---

### Step 2 — Data Adaptation Layer

PSCL was built for a different dataset (376×376 RGB images, 4 classes one-hot encoded). A new dataset class `data_metaldam.py` bridges the gap on the fly so the original MetalDAM patches are never modified on disk.

**2a. Grayscale → RGB**

SEM images are single-channel. PSCL's encoder expects 3 channels. Replicate the channel three times:

```python
img_rgb = np.repeat(img[np.newaxis], 3, axis=0)  # shape: (3, H, W)
```

This preserves pixel values without adding any new information.

**2b. Resize 256×256 → 376×376**

- **Images:** `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` — smooth interpolation for intensity data.
- **Masks:** `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` — this is mandatory. Any other mode blends integer class indices and creates phantom classes along boundaries, corrupting training labels.

**2c. ImageNet normalisation**

PSCL's pipeline normalises with ImageNet statistics:

```python
mean = (0.49139968, 0.48215841, 0.44653091)
std  = (0.24703223, 0.24348513, 0.26158784)
```

Apply channel-wise after the grayscale replication.

**2d. Class remapping**

MetalDAM has 5 classes; PSCL's UNet architecture is hardcoded for 4 output channels. The fix is to treat MetalDAM's Background/Defect class (index 0) as the ignore label instead of a foreground class, then shift the 4 real phases down by one:

| MetalDAM index | Phase | PSCL index |
|----------------|-------|------------|
| 0 | Background / Defect | **255 (ignore)** |
| 1 | Austenite | 0 |
| 2 | Matrix | 1 |
| 3 | Martensite-Austenite (MA) | 2 |
| 4 | Precipitate | 3 |

```python
REMAP = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3}
mask_remapped = np.vectorize(REMAP.get)(mask)
```

No changes to `model.py` are needed — the 4-class UNet architecture stays exactly as-is.

**2e. One-hot encoding**

PSCL's dataset classes return masks as `(B, 4, H, W)` float32 one-hot tensors. Reuse the existing `get_one_hot(mask, num_classes=4)` utility already present in `PSCL/data.py`. Pixels carrying label 255 should be zeroed across all 4 channels so the loss ignores them.

---

### Step 3 — PSCL Config Changes

Only `config.py` needs path and hyperparameter updates — no model changes:

```python
# config.py  (inside runner_preheat)

data_dir = "/mnt/e/Afile/1-Python/MetalDam/data/patches"

# Learning rates — separate for encoder (pretrained) and decoder (fresh)
finetune_lr_en = 1e-4
finetune_lr_de = 1e-3

batch_size     = 8
finetune_epoch = 100

# Class weights — upweight rare phases (MA and Precipitate are sparse)
weight      = [1, 1, 5, 5]   # BCE weights per class
dice_weight = [1, 1, 1, 1]   # Dice weights per class
```

In `config_run.py`, verify the Dice and BCE loss weight vectors have length 4 — they already do for the default setup.

---

### Step 4a — Path A: PSCL Self-Supervised Pretraining (recommended)

PSCL's core contribution is **patch-sampled contrastive learning (PSCL)**: it uses a small labelled sample to guide which patch pairs are semantically similar during self-supervised pretraining, producing a better-initialised encoder for downstream segmentation.

**How to run:**

1. Use the MetalDAM train split (788 patches) as the **unlabeled pool**.
2. Hold out a tiny subset (~1 parent micrograph, ~20–30 patches) as the **labelled guidance sample** — this mimics the few-shot regime PSCL was designed for.
3. Run pretraining:
   ```bash
   python config_run.py --mode pretrain --method moco   # or simclr
   ```
4. The loss combines:
   - **Global loss:** image-level InfoNCE contrastive loss
   - **Dense loss:** multi-scale patch-level contrastive loss, guided by the labelled sample

This produces a pretrained `UNet_encode` checkpoint with domain-adapted feature representations of steel microstructure.

---

### Step 4b — Path B: Direct Finetuning (ablation baseline)

Skip pretraining entirely. Load either ImageNet-pretrained or randomly initialised encoder weights and go straight to supervised finetuning. This path exists to **measure how much the self-supervised pretraining stage actually helps** on this domain — if Path A's test mIoU is significantly higher, the PSCL pretraining is worth the compute cost.

```bash
python config_run.py --mode supervised   # full supervised baseline
```

---

### Step 5 — Finetuning

Regardless of Path A or B, finetuning follows the same script:

```bash
python config_run.py --mode finetune
```

What happens inside `config_run.py → Finetune()`:

1. Load the pretrained `UNet_encode` weights from the checkpoint saved in Step 4a (or ImageNet init for Path B).
2. Attach a fresh `UNet_decode` (4 output channels, randomly initialised).
3. Optionally **freeze the encoder** for the first ~10 epochs so the decoder stabilises before end-to-end training begins.
4. Optimise with `AdamW`, using separate learning rates:
   - Encoder: `finetune_lr_en = 1e-4` (smaller — preserves pretrained features)
   - Decoder: `finetune_lr_de = 1e-3` (larger — trains from scratch)
5. Combined segmentation loss per batch:
   ```
   loss = (1 - dice_bce_ratio) × BCE_loss + dice_bce_ratio × Dice_loss
   ```
   BCE class weights upweight rare phases (MA, Precipitate).
   Pixels with remapped label 255 are excluded from both loss terms.
6. After each epoch, evaluate on the val split (168 patches) and **save the checkpoint whenever val mIoU improves**.

---

### Step 6 — Evaluation

After finetuning, run final evaluation on the held-out test split (192 patches — never seen during training or model selection):

```bash
python config_run.py --mode test
```

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across all 4 foreground classes |
| **IoU per class** | Austenite, Matrix, MA, Precipitate individually |
| **Dice coefficient** | Harmonic mean of precision and recall per class |
| **Pixel accuracy** | Fraction of correctly classified pixels |

Compare Path A (PSCL pretrained encoder) vs Path B (direct finetuning) on these metrics to confirm the self-supervised pretraining stage adds value for steel microstructure segmentation.

---

## Files to Create / Modify

| Action | File | Purpose |
|--------|------|---------|
| **Create** | `Pscl_finetune/data_metaldam.py` | Dataset adapter (steps 2a–2e) |
| **Modify** | `PSCL/PSCL/config.py` | `data_dir`, learning rates, class weights |
| **Modify** | `PSCL/PSCL/config_run.py` | Wire in the new dataset class |
| No change | `PSCL/PSCL/model.py` | 4-class UNet architecture unchanged |
| No change | `PSCL/PSCL/data.py` | `get_one_hot` reused as-is |

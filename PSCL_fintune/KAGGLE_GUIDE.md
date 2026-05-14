# Running MetalDAM → PSCL Training on Kaggle (Script Mode)

Training is driven by a single Python script (`kaggle_run.py`). A one-cell Kaggle Script
clones the repo and calls the script; all training logic stays outside the notebook cells.

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Kaggle account | Phone-verified (required to unlock GPU and internet) |
| Internet enabled | Notebook Settings → Internet → **On** (needed to clone from GitHub) |
| GPU enabled | Notebook Settings → Accelerator → **GPU T4 x2** |

No Kaggle CLI, no dataset upload, no zip files needed. The MetalDAM patches are already
in the GitHub repo and are cloned automatically.

---

## Step 1 — Create a Kaggle Script

1. Go to `kaggle.com/code` → **New Notebook**
2. In the editor: **File → Script** to switch from notebook to script mode
3. In the right panel → Settings:
   - **Accelerator**: GPU T4 x2 (or P100)
   - **Internet**: On
4. Delete any default content in the editor

---

## Step 2 — Paste the launcher

Copy the block below into the Kaggle Script editor. Edit only the variables at the top.

```python
import os, subprocess, sys

# ── Edit these ────────────────────────────────────────────────────────────────
PSCL_REPO        = 'https://github.com/neulmc/PSCL'      # original PSCL repo
DATA_DIR         = '/kaggle/working/repo/MetalDam/data/patches'
PRETRAIN_EPOCHS  = 100      # 200 for a full run
RESUME_DATASET   = ''       # '/kaggle/input/pscl-checkpoints' to resume
STAGE            = 'both'   # 'self' | 'fine' | 'both'
SELF_LR          = 2e-3     # Stage 1 pretraining LR
FINETUNE_LR_EN   = 1e-3     # Stage 2 encoder LR
FINETUNE_LR_DE   = 1e-3     # Stage 2 decoder LR
# ─────────────────────────────────────────────────────────────────────────────

BASE = '/kaggle/input/pscl-files'
repo = '/kaggle/working/repo'

if os.path.exists(repo):
    subprocess.run(['git', '-C', repo, 'pull'], check=True)
else:
    subprocess.run(['git', 'clone', '--branch', 'kaggle',
                    'https://github.com/arhorri/PSCL_fintune.git', repo], check=True)

subprocess.run([
    sys.executable,
    f'{repo}/PSCL_fintune/kaggle_run.py',
    f'--base={BASE}',
    f'--pscl-repo={PSCL_REPO}',
    f'--data-dir={DATA_DIR}',
    f'--pretrain-epochs={PRETRAIN_EPOCHS}',
    f'--resume-dataset={RESUME_DATASET}',
    f'--stage={STAGE}',
    f'--self-lr={SELF_LR}',
    f'--finetune-lr-en={FINETUNE_LR_EN}',
    f'--finetune-lr-de={FINETUNE_LR_DE}',
], check=True)
```

Click **Save Version → Save & Run All (Commit)**.

---

## Step 3 — What the script does

**Stage 1 — Self-supervised pretraining (MoCo)**
- Unlabeled pool: train split (788 patches)
- Labeled guidance: val split (168 patches) — guides patch sampling only
- Saves a checkpoint every 10 epochs to `self_UNet_metaldam/_Numf/f/moco{N}.pt`
- Estimated time: ~6–9 h for 200 epochs on a T4

**Stage 2 — Supervised finetuning**
- Loads `moco{N}.pt`, attaches a fresh 4-class decoder
- Separate LRs: encoder (`FINETUNE_LR_EN`) and decoder (`FINETUNE_LR_DE`)
- Validates on val split; saves best model as `fine_UNet_metaldam/_Numf/f/fine.pt`
- Prints final test-split metrics: mIoU, per-class IoU, Dice, pixel accuracy
- Estimated time: ~1 h for 50 epochs on a T4

---

## Step 4 — Checkpoint resumption (multi-session)

Kaggle sessions last at most ~9 hours. Stage 1 (200 epochs) will span multiple sessions.

### After a session ends

1. Open the **Output** tab of the finished run
2. Download all `moco*.pt` files from `self_UNet_metaldam/_Numf/f/`

### Upload checkpoints as a new Kaggle dataset

1. Go to **kaggle.com → Datasets → New Dataset**, name it `pscl-checkpoints`
2. Create the folder structure locally and upload:

```bash
mkdir -p checkpoints_upload/self_UNet_metaldam/_Numf/f
cp moco*.pt checkpoints_upload/self_UNet_metaldam/_Numf/f/
```
Then drag the `checkpoints_upload/` folder into the Kaggle dataset editor and save.

### In the next session

Attach `arhorri/pscl-checkpoints` via **Add Data**, then update the launcher:

```python
RESUME_DATASET  = '/kaggle/input/pscl-checkpoints'
STAGE           = 'self'
PRETRAIN_EPOCHS = 200    # same target; script auto-detects latest checkpoint and resumes
```

### Starting Stage 2 after pretraining is complete

```python
RESUME_DATASET  = '/kaggle/input/pscl-checkpoints'  # restore moco200.pt
STAGE           = 'fine'
PRETRAIN_EPOCHS = 200
```

---

## Step 5 — Outputs

All files appear in the **Output** tab:

| File | Description |
|------|-------------|
| `self_UNet_metaldam/_Numf/f/moco{N}.pt` | Stage 1 checkpoints (every 10 epochs) |
| `self_UNet_metaldam/_Numf/f/log_self.txt` | Pretraining loss log |
| `fine_UNet_metaldam/_Numf/f/fine.pt` | Best finetuned model (by val accuracy) |
| `fine_UNet_metaldam/_Numf/f/log_fine.txt` | Finetuning loss + val metrics log |
| `training_curves.png` | Loss and val metrics plot |

---

## CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | (computed from `--base`) | Direct path to `MetalDam/data/patches/` |
| `--pscl-repo` | (none) | GitHub URL of PSCL repo to clone |
| `--pretrain-epochs` | `200` | Total Stage 1 epochs |
| `--self-lr` | `1e-4` | Stage 1 pretraining learning rate |
| `--finetune-lr-en` | `1e-4` | Stage 2 encoder LR |
| `--finetune-lr-de` | `1e-3` | Stage 2 decoder LR |
| `--resume-dataset` | `''` | Dataset path containing prior `moco*.pt` files |
| `--stage` | `both` | `self` / `fine` / `both` |
| `--gpu` | `0` | CUDA device index |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Could not resolve host: github.com` | Internet is off | Notebook Settings → Internet → On |
| `fatal: destination path already exists` | Repo left from previous run | Already handled — launcher does `git pull` if repo exists |
| `could not read Username` | PSCL repo is private or doesn't exist | Use `PSCL_REPO = 'https://github.com/neulmc/PSCL'` |
| `Data verification FAILED` | Wrong `DATA_DIR` | Set `DATA_DIR = '/kaggle/working/repo/MetalDam/data/patches'` |
| `No module named 'model'` | PSCL model.py not on sys.path | Script auto-detects; ensure `--pscl-repo` points to a valid PSCL repo |
| `Stage 1 checkpoint not found` | Running `fine` without Stage 1 done | Set `RESUME_DATASET` pointing to a dataset with `moco{N}.pt` |
| `No GPU detected` | Accelerator not enabled | Notebook Settings → Accelerator → GPU T4 |

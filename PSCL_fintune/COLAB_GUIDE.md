# Running MetalDAM → PSCL Training on Google Colab

Training is driven by `colab_run.py`. Three notebook cells clone the repo, run the
script, and save checkpoints automatically to Google Drive — no manual downloads needed.

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Google account | For Colab and Google Drive |
| Google Drive | Used to persist checkpoints between sessions |
| GPU runtime | Runtime → Change runtime type → **GPU (T4)** |

No CLI tools, no zip files, no dataset uploads needed. The MetalDAM patches are already
in the GitHub repo and are cloned automatically.

---

## Step 1 — Open Colab and enable GPU

1. Go to **colab.research.google.com** → **New notebook**
2. **Runtime → Change runtime type**
3. Set **Hardware accelerator** → **GPU** → Save

---

## Step 2 — Create 3 cells

Delete any default content. Create 3 cells and paste the following.

### Cell 1 — Config (the only cell you edit each session)

```python
DRIVE_DIR        = '/content/drive/MyDrive/PSCL_training'  # folder in your Drive
PSCL_REPO        = 'https://github.com/neulmc/PSCL'
PRETRAIN_EPOCHS  = 200
STAGE            = 'both'   # 'self' | 'fine' | 'both'
SELF_LR          = 1e-3     # Stage 1 pretraining LR
FINETUNE_LR_EN   = 1e-3     # Stage 2 encoder LR
FINETUNE_LR_DE   = 1e-3     # Stage 2 decoder LR
```

### Cell 2 — Clone repo (runs once per session, safe to re-run)

```python
import os, subprocess, sys

repo = '/content/repo'
if os.path.exists(repo):
    subprocess.run(['git', '-C', repo, 'pull'], check=True)
else:
    subprocess.run(['git', 'clone', '--branch', 'kaggle',
                    'https://github.com/arhorri/PSCL_fintune.git', repo], check=True)
print('Repo ready.')
```

### Cell 3 — Run training

```python
subprocess.run([
    sys.executable,
    f'{repo}/PSCL_fintune/colab_run.py',
    f'--drive-dir={DRIVE_DIR}',
    f'--pscl-repo={PSCL_REPO}',
    f'--pretrain-epochs={PRETRAIN_EPOCHS}',
    f'--stage={STAGE}',
    f'--self-lr={SELF_LR}',
    f'--finetune-lr-en={FINETUNE_LR_EN}',
    f'--finetune-lr-de={FINETUNE_LR_DE}',
], check=True)
```

Click **Runtime → Run all**.

---

## Step 3 — What the script does

On first run the script:
1. Mounts Google Drive (a browser popup asks for permission — click Allow)
2. Clones PSCL source from GitHub
3. Verifies the 1,148 MetalDAM patches from the cloned repo
4. Runs Stage 1 and/or Stage 2
5. **Auto-saves all outputs to `DRIVE_DIR` when done**

**Stage 1 — Self-supervised pretraining (MoCo)**
- Unlabeled pool: train split (788 patches)
- Labeled guidance: val split (168 patches) — guides patch sampling only
- LR schedule: `SELF_LR` → ×0.1 at 50% epochs → ×0.1 at 75% epochs
- Saves a checkpoint every 10 epochs to `self_UNet_metaldam/_Numf/f/moco{N}.pt`
- Estimated time: ~6–9 h for 200 epochs on a T4

**Stage 2 — Supervised finetuning**
- Loads `moco{N}.pt`, attaches a fresh 4-class decoder
- Separate LRs: encoder (`FINETUNE_LR_EN`) and decoder (`FINETUNE_LR_DE`)
- Validates on val split every `test_freq` epochs; saves best model as `fine.pt`
- Prints final test-split metrics: mIoU, per-class IoU, Dice, pixel accuracy
- Estimated time: ~1 h for 50 epochs on a T4

---

## Step 4 — Checkpoint resumption (multi-session)

Colab free tier sessions last ~12 hours; Pro ~24 hours. Stage 1 (200 epochs, ~12 h)
may need two sessions. Checkpoints are **saved to Drive automatically** at the end of
each run, so resumption is simple.

### In the next session

Just run all 3 cells again — no changes needed. The script:
1. Mounts Drive
2. Restores the latest `moco*.pt` from `DRIVE_DIR`
3. Continues from the last saved epoch

### Running only Stage 1 (to save time for Stage 2 later)

```python
STAGE            = 'self'
PRETRAIN_EPOCHS  = 200
```

### Running only Stage 2 after pretraining is complete

```python
STAGE            = 'fine'
PRETRAIN_EPOCHS  = 200   # must match the epoch used in Stage 1
```

---

## Step 5 — Outputs

All outputs are saved automatically to `DRIVE_DIR/` in your Google Drive:

| File | Description |
|------|-------------|
| `self_UNet_metaldam/_Numf/f/moco{N}.pt` | Stage 1 checkpoints (every 10 epochs) |
| `self_UNet_metaldam/_Numf/f/log_self.txt` | Pretraining loss log |
| `fine_UNet_metaldam/_Numf/f/fine.pt` | Best finetuned model (by val accuracy) |
| `fine_UNet_metaldam/_Numf/f/log_fine.txt` | Finetuning loss + val metrics log |
| `training_curves.png` | Loss and val metrics plot |

You can also find them in Colab's **Files panel** (left sidebar) under `/content/repo/PSCL_fintune/`.

---

## CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--drive-dir` | `''` | Google Drive folder for checkpoint persistence |
| `--pscl-repo` | `https://github.com/neulmc/PSCL` | GitHub URL of PSCL repo |
| `--data-dir` | `/content/repo/MetalDam/data/patches` | Path to MetalDam patches |
| `--pretrain-epochs` | `200` | Total Stage 1 epochs |
| `--self-lr` | `1e-3` | Stage 1 pretraining LR |
| `--finetune-lr-en` | `1e-3` | Stage 2 encoder LR |
| `--finetune-lr-de` | `1e-3` | Stage 2 decoder LR |
| `--stage` | `both` | `self` / `fine` / `both` |
| `--gpu` | `0` | CUDA device index |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No GPU detected` | GPU runtime not enabled | Runtime → Change runtime type → GPU |
| Drive mount popup doesn't appear | Pop-ups blocked | Allow pop-ups for colab.research.google.com |
| `Data verification FAILED` | MetalDam not in repo | Make sure Cell 2 ran successfully and repo cloned to `/content/repo` |
| `No module named 'model'` | PSCL clone failed or wrong structure | Check `PSCL_REPO` URL; script auto-detects `model.py` location |
| `Stage 1 checkpoint not found` | Running `fine` before Stage 1 | Set `STAGE = 'self'` first, or ensure `moco{N}.pt` exists in Drive |
| Session disconnected mid-training | Colab idle timeout | Checkpoints are saved every 10 epochs; resume by re-running all cells |
| `could not read Username` | PSCL repo is private | Use `PSCL_REPO = 'https://github.com/neulmc/PSCL'` (public repo) |

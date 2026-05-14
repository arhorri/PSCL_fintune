# Pscl_fintune

Adapter layer that connects MetalDAM preprocessed patches to PSCL finetuning.
See `FINETUNING_GUIDE.md` for the full flowchart and step-by-step explanation.

---

## Repository

| | |
|---|---|
| GitHub | `https://github.com/arhorri/PSCL_fintune` |
| SSH remote | `git@github.com:arhorri/PSCL_fintune.git` |
| Active branch | `kaggle` (Kaggle script mode) |

---

## Data paths

| Dataset | Path |
|---------|------|
| MetalDAM patches (images) | `../MetalDam/data/patches/{train,val,test}/images_norm/` |
| MetalDAM patches (masks) | `../MetalDam/data/patches/{train,val,test}/masks/` |
| PSCL source code | `../PSCL/PSCL/` |
| PSCL config | `../PSCL/PSCL/config.py` |
| PSCL training entry | `../PSCL/PSCL/config_run.py` |

Split counts: train 788 patches / val 168 / test 192 (split at parent-image level in MetalDAM).

---

## Files to create / modify

| Action | File | Purpose |
|--------|------|---------|
| **Create** | `Pscl_fintune/data_metaldam.py` | Dataset adapter — bridges format differences |
| **Modify** | `../PSCL/PSCL/config.py` | `data_dir`, learning rates, class weights |
| **Modify** | `../PSCL/PSCL/config_run.py` | Wire in `data_metaldam.py` for data loading |
| No change | `../PSCL/PSCL/model.py` | 4-class UNet architecture unchanged |
| No change | `../PSCL/PSCL/data.py` | `get_one_hot` reused as-is |

---

## Commands

```bash
# Path A — self-supervised pretraining first (recommended)
cd ../PSCL/PSCL
python config_run.py --mode pretrain --method moco   # or simclr

# Finetuning (both paths)
python config_run.py --mode finetune

# Path B — supervised baseline (ablation, skip pretraining)
python config_run.py --mode supervised

# Evaluation on held-out test split
python config_run.py --mode test
```

---

## Key invariants

The rules in `.claude/rules/` encode the non-obvious constraints that cause silent bugs if violated.
The most critical:

- Mask resize → `INTER_NEAREST` only (`rules/mask-interpolation.md`)
- Class remap before one-hot encoding (`rules/class-remapping.md`)
- ImageNet normalisation applied **after** channel replication (`rules/image-normalisation.md`)
- Test split is never used for model selection (`rules/evaluation-protocol.md`)

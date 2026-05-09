---
description: Enforce held-out test split and required metrics for PSCL finetuning evaluation
alwaysApply: true
---

The test split (192 patches) is held out and must never be used for model selection or hyperparameter tuning.

- **Model selection:** save checkpoint based on **val mIoU** (168 patches) after each epoch.
- **Final evaluation:** run once on the test split after all training is complete.

Required metrics (report all four):

| Metric | Scope |
|--------|-------|
| mIoU | Mean across all 4 foreground classes |
| Per-class IoU | Austenite (0), Matrix (1), MA (2), Precipitate (3) separately |
| Dice coefficient | Per class |
| Pixel accuracy | Overall |

Always run evaluation for **both** Path A (PSCL pretrained) and Path B (direct finetune / supervised baseline) and report side-by-side. Without Path B as a reference, there is no way to know whether the self-supervised pretraining stage contributed anything on this domain.

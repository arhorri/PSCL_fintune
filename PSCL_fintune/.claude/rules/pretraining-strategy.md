---
description: Guidance on choosing between PSCL self-supervised pretraining (Path A) and direct finetuning (Path B)
alwaysApply: false
---

**Path A — PSCL self-supervised pretraining (recommended)**

Use the MetalDAM train split (788 patches) as the unlabeled pool. Hold out patches from one parent micrograph (~20–30 patches) as the labelled guidance sample — this small labeled set guides PSCL's patch-sampling during contrastive pretraining without consuming supervision budget.

Run with MoCo (preferred — uses momentum encoder + memory queue, more stable) or SimCLR.

The pretrained `UNet_encode` checkpoint is then loaded for finetuning (Step 5 in the guide).

**Path B — direct finetuning (required ablation)**

Skip pretraining. Load ImageNet-initialised or randomly initialised encoder weights and go straight to supervised finetuning with the full train split.

Path B is not optional — it is the required baseline that measures the value of pretraining. If Path A and Path B produce similar test mIoU, the pretraining stage is not helping on this domain and should be investigated.

---
description: Enforce separate learning rates for encoder and decoder during PSCL finetuning
alwaysApply: true
---

Always use two separate parameter groups in the AdamW optimizer — one for the encoder, one for the decoder:

```python
optimizer = AdamW([
    {"params": model.encode.parameters(), "lr": 1e-4},  # encoder — pretrained, lower LR
    {"params": model.decode.parameters(), "lr": 1e-3},  # decoder — fresh, higher LR
], weight_decay=1e-4)
```

Never use a single learning rate for the whole model. The encoder carries pretrained PSCL features that would be destroyed by a decoder-scale LR. The decoder is randomly initialised and needs a higher LR to converge in the same number of epochs.

Config values in `PSCL/PSCL/config.py`: `finetune_lr_en = 1e-4`, `finetune_lr_de = 1e-3`.

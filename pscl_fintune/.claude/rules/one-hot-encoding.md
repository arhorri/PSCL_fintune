---
description: Enforce correct one-hot encoding of remapped masks for PSCL input
alwaysApply: true
---

Convert remapped masks to one-hot using the existing utility in `PSCL/PSCL/data.py`:

```python
from data import get_one_hot
mask_onehot = get_one_hot(mask_remapped, num_classes=4)  # (4, H, W) float32
```

After encoding, zero out all channels for pixels that carry the ignore label (255):

```python
ignore = (mask_remapped == 255)
mask_onehot[:, ignore] = 0.0
```

Do not rewrite `get_one_hot` — reuse it as-is. The zeroing step is critical: without it, ignored pixels contribute a valid one-hot vector to the loss, effectively training the model on pixels that should be masked out.

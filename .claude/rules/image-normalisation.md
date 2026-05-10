---
description: Enforce correct normalisation order and statistics for MetalDAM images entering PSCL
alwaysApply: true
---

Normalise images with ImageNet statistics **after** channel replication, never before.

```python
IMAGENET_MEAN = (0.49139968, 0.48215841, 0.44653091)
IMAGENET_STD  = (0.24703223, 0.24348513, 0.26158784)
```

Order of operations in `data_metaldam.py`:
1. Load uint8 PNG → divide by 255 → float32 `[0, 1]`
2. Repeat channel ×3
3. Apply ImageNet mean/std channel-wise

If normalisation is applied before replication (to the single-channel image), the per-channel statistics are wrong and the three replicated channels will be identically off-scale, shifting every activation in the encoder.

MetalDAM's own z-score normalisation is already baked into the `images_norm/` PNGs. Step 1 (÷255) undoes the uint8 encoding; the ImageNet normalisation then aligns the value range with what PSCL's encoder expects.

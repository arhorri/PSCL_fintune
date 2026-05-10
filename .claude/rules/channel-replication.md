---
description: Enforce correct grayscale-to-RGB conversion for MetalDAM images before passing to PSCL
alwaysApply: true
---

Convert grayscale MetalDAM images to 3-channel by repeating the single channel:

```python
img_rgb = np.repeat(img[np.newaxis], 3, axis=0)  # (1,H,W) -> (3,H,W)
```

Never use `cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)` — OpenCV reorders axes to `(H,W,3)` and requires an additional transpose, which is error-prone. The repeat approach keeps the `(C,H,W)` tensor layout expected by PyTorch throughout.

All three channels carry identical values. This is intentional: SEM images are physically single-channel; replication lets the ImageNet-pretrained encoder process them without altering the feature space.

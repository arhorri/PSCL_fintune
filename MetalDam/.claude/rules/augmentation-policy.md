---
description: Augmentation safety policy — which transforms are safe vs banned for SEM micrograph/mask pairs
globs: src/**/*.py
---

Apply augmentations with `albumentations`. Identical spatial transforms must be applied to both image and mask.

| Transform | Status | Reason |
|-----------|--------|--------|
| HorizontalFlip | safe | |
| VerticalFlip | safe | |
| RandomRotate90 | safe | |
| GaussNoise | safe | |
| GaussianBlur (kernel ≤ 3) | safe | |
| RandomBrightnessContrast (±10 %) | light | |
| ElasticTransform | **banned** | alters grain morphology |
| ShiftScaleRotate (scale ≠ 1) | **banned** | breaks µm/px calibration |
| Heavy colour jitter | skip | SEM images are effectively grayscale |

---
description: Enforce INTER_NEAREST interpolation for all mask/label resizes to prevent phantom class creation
alwaysApply: true
---

Resize masks with `cv2.INTER_NEAREST` only. Never use `INTER_LINEAR`, `INTER_CUBIC`, or `INTER_AREA` on class index masks. Interpolating class labels creates phantom classes along boundaries that corrupt segmentation training.

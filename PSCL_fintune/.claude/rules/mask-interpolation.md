---
description: Enforce INTER_NEAREST for all mask resizes in the data adapter to prevent phantom class creation
alwaysApply: true
---

Resize masks with `cv2.INTER_NEAREST` only. Never use `INTER_LINEAR`, `INTER_CUBIC`, or `INTER_AREA` on class-index masks.

Any other interpolation blends integer class indices at boundaries, creating values that correspond to no real class. This silently corrupts training labels — the model receives phantom classes it can never learn to predict.

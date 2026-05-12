---
description: Enforce using label.shape[0] as the authoritative crop height when aligning SEM images to masks
globs: src/preprocess/crop_align.py
---

Crop the micrograph to `label.shape[0]`, not to an assumed row. Labels are already bar-free. Use `label.shape[0]` as the authoritative crop height. Always assert `img_cropped.shape[:2] == mask.shape[:2]` before saving.

---
description: Enforce that scale extraction runs before any cropping step in the preprocessing pipeline
alwaysApply: true
---

Extract scale BEFORE cropping. `extract_scale.py` must run before `crop_align.py` on every image. Never crop first. The info bar is the only source of µm/pixel ground truth — once cropped away it cannot be recovered.

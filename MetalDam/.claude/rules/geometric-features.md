---
description: Downstream geometric feature extraction — enforce um_per_pixel scaling on all regionprops measurements
globs: src/analysis/geometric_features.py
---

Every `skimage.measure.regionprops` measurement must be multiplied by `um_per_pixel` from `metadata.csv`. Without this, grain diameter values are pixel counts, not micrometres.

Key outputs per image:
- Mean equivalent grain diameter (µm)
- Phase fraction per class
- Mean aspect ratio
- Boundary density (µm⁻¹)

Feed these as a feature vector into a regression model targeting hardness / yield strength.

---
description: RGB-to-class-index mapping for MetalDAM label encoding
globs: src/preprocess/label_encoding.py
---

Verify this map against the MetalDAM README before the first run.

| RGB | Class index | Phase |
|-----|-------------|-------|
| (0, 255, 0) | 1 | Austenite |
| (128, 0, 255) | 2 | Matrix |
| (255, 255, 0) | 3 | Martensite-Austenite (MA) |
| (255, 0, 0) | 4 | Precipitate |
| (255, 0, 255) | 0 | Background / Defect |

Tolerance for JPEG colour bleeding: L2 < 20. If any class covers < 0.5 % of an image, `validate_color_map()` will warn.

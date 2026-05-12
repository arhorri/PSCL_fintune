---
description: metadata.csv column schema — every preprocessing step appends here; columns must never be removed
globs: src/**/*.py
---

Every preprocessing step appends to `metadata.csv`. Never delete columns.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | patch filename |
| `parent_id` | str | stem of the source SEM image |
| `split` | str | train / val / test |
| `um_per_pixel` | float | µm per pixel **after** resolution normalisation |
| `magnification_kx` | float | from SEM info bar |
| `kV` | float | accelerating voltage |
| `height` | int | patch height px |
| `width` | int | patch width px |
| `patch_y` | int | top-left y in parent image |
| `patch_x` | int | top-left x in parent image |
| `class_histogram` | JSON str | `{"0": n, "1": n, ...}` |

Do not commit `metadata.csv` with missing `um_per_pixel` values.

---
description: Enforce the MetalDAM-to-PSCL class index mapping in the data adapter
alwaysApply: true
---

Apply this exact remapping to every mask before one-hot encoding. Never pass raw MetalDAM indices to PSCL.

| MetalDAM index | Phase | PSCL index |
|----------------|-------|------------|
| 0 | Background / Defect | **255 (ignore)** |
| 1 | Austenite | 0 |
| 2 | Matrix | 1 |
| 3 | Martensite-Austenite (MA) | 2 |
| 4 | Precipitate | 3 |

```python
REMAP = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3}
mask = np.vectorize(REMAP.get)(mask)
```

Background is mapped to ignore (255) — not class 0 — so that PSCL's 4-class architecture requires no changes. Pixels with value 255 must be excluded from both Dice and BCE loss terms.

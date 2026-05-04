---
description: Enforce train/val/test splitting at the parent image level to prevent data leakage
alwaysApply: true
---

Split at the PARENT IMAGE level, never the patch level. All patches from the same parent must land in the same split. Patch-level splitting causes data leakage and inflates validation metrics.

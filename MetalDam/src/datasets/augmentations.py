"""
Albumentations augmentation pipelines for SEM micrograph segmentation.

Policy (from augmentation-policy.md):
  SAFE   — HorizontalFlip, VerticalFlip, RandomRotate90, GaussNoise,
            GaussianBlur (kernel ≤ 3), RandomBrightnessContrast (±10 %)
  BANNED — ElasticTransform, ShiftScaleRotate with scale≠1, heavy colour jitter

All spatial transforms are applied identically to both image and mask by
passing additional_targets={"mask": "mask"} at the Compose level.
Albumentations guarantees the same random parameters are used for both
when the transform receives image+mask in a single call.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── excluded transforms — rationale ──────────────────────────────────────────
#
# ElasticTransform
#   Warps the image with a random displacement field.  For SEM micrographs
#   this distorts actual grain boundaries and phase morphologies, making
#   the image physically impossible.  Geometric features (grain diameter,
#   aspect ratio) computed from the mask would no longer match the image,
#   invalidating any downstream regression on hardness / yield strength.
#
# ShiftScaleRotate (scale != 1.0)
#   Scaling the image changes the number of pixels per µm without updating
#   um_per_pixel in metadata.  Grain size measurements would be off by the
#   scale factor, silently corrupting the geometric feature pipeline.
#   Rotation-only (scale=1) would be safe, but RandomRotate90 already
#   covers the relevant symmetries for SEM images.
#
# Heavy colour jitter (Hue/Saturation, RGBShift, large brightness swings)
#   SEM backscattered / secondary electron images are effectively grayscale.
#   Large colour shifts do not produce realistic variations and can push
#   pixel intensities outside the range seen at inference, hurting
#   generalisation.  Small brightness/contrast variation (±10 %) is kept
#   because detector gain and beam current vary slightly between sessions.
#
# ─────────────────────────────────────────────────────────────────────────────


def get_train_transforms(p_flip: float = 0.5) -> A.Compose:
    """Return the training augmentation pipeline.

    Spatial transforms are applied to both ``image`` and ``mask`` via
    ``additional_targets``.  Pixel-level transforms (noise, blur,
    brightness) are applied to ``image`` only — masks are class indices
    and must not be altered by intensity operations.

    Parameters
    ----------
    p_flip:
        Probability for each flip / rotate transform.  Default 0.5.

    Returns
    -------
    A.Compose
        Call with ``transform(image=img, mask=mask)`` where both arrays
        are H×W uint8 numpy arrays.

    Examples
    --------
    >>> transforms = get_train_transforms()
    >>> result = transforms(image=img_uint8, mask=mask_uint8)
    >>> aug_img, aug_mask = result["image"], result["mask"]
    """
    return A.Compose(
        [
            # ── spatial (applied to image AND mask) ───────────────────────
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_flip),

            # ── pixel-level (image only — mask is passed through unchanged) ─
            # GaussianBlur: kernel must be odd and ≤ 3 (policy limit).
            # Simulates slight defocus variation between SEM sessions.
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),

            # GaussNoise: simulates detector shot noise.
            A.GaussNoise(p=0.3),

            # RandomBrightnessContrast: ±10 % only.
            # Accounts for minor beam-current and detector-gain drift.
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.3,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transforms() -> A.Compose:
    """Return the validation / test augmentation pipeline (identity only).

    No transforms are applied.  The pipeline is provided as a no-op so
    training and evaluation code paths are structurally identical and
    ``additional_targets`` is consistently defined.

    Returns
    -------
    A.Compose
        Call with ``transform(image=img, mask=mask)``.
    """
    return A.Compose(
        [],
        additional_targets={"mask": "mask"},
    )


def get_tensor_transforms() -> A.Compose:
    """Return a normalisation + to-tensor pipeline for use after augmentation.

    Converts a float32 [0, 1] numpy image to a (C, H, W) torch.Tensor.
    Must be applied AFTER ``get_train_transforms`` / ``get_val_transforms``
    and AFTER intensity normalisation (Node 07).

    Note: ``ToTensorV2`` does NOT normalise pixel values — that is done in
    Node 07 (intensity_norm.py) so the tensors reflect the chosen method
    (zscore / clahe / minmax) rather than a hard-coded ImageNet mean.

    Returns
    -------
    A.Compose
        Call with ``transform(image=float32_img)`` (mask excluded here;
        mask conversion to LongTensor is handled in the Dataset class).
    """
    return A.Compose([ToTensorV2()])


# ── usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    import numpy as np

    # Synthetic H×W uint8 arrays matching a 256-px patch.
    rng = np.random.default_rng(0)
    img_uint8  = rng.integers(80, 200, (256, 256), dtype=np.uint8)
    mask_uint8 = rng.integers(0, 5,   (256, 256), dtype=np.uint8)

    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    result_train = train_tf(image=img_uint8, mask=mask_uint8)
    result_val   = val_tf(image=img_uint8,  mask=mask_uint8)

    aug_img  = result_train["image"]
    aug_mask = result_train["mask"]

    print("=== Train transforms ===")
    print(f"image  shape: {aug_img.shape}   dtype: {aug_img.dtype}")
    print(f"mask   shape: {aug_mask.shape}  dtype: {aug_mask.dtype}")
    print(f"Unique mask values (must match input classes): {np.unique(aug_mask).tolist()}")
    assert aug_img.shape == img_uint8.shape,  "image shape changed"
    assert aug_mask.shape == mask_uint8.shape, "mask shape changed"
    assert set(np.unique(aug_mask)).issubset(set(np.unique(mask_uint8))), \
        "augmentation introduced new class values in mask"

    print("\n=== Val transforms (identity) ===")
    assert np.array_equal(result_val["image"], img_uint8),  "val image was modified"
    assert np.array_equal(result_val["mask"],  mask_uint8), "val mask was modified"
    print("Identity confirmed — image and mask unchanged.")

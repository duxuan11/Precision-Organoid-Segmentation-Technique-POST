"""Contrast enhancement for the TinySAM encoder input.

Ported from cellSAM's ``cellSAM/utils.py`` (``_percentile_threshold`` /
``_histogram_normalization`` / ``enhance_low_contrast``) but implemented with numpy + OpenCV
so POST does not gain a scikit-image dependency.

Only the **encoder** sees the enhanced image. The overlay produced by ``show_anns`` keeps using
the untouched source image, so the visualisation still shows what was acquired.

Modes (``TinySam(enhance=...)`` / ``inference.py --enhance``):

``none``
    No-op, the historical behaviour. The TinySAM ONNX encoder was fine-tuned with a fixed
    mean/std normalisation, so this stays the default until it is validated on real weights.
``clahe``
    Percentile clipping (kills specular highlights such as bubbles) followed by CLAHE.
``lowcontrast``
    Same pipeline, but only applied when the image actually looks low contrast. The gate is a
    simplification of cellSAM's ``0.04 < mean_diff < 0.05`` window: CLAHE is applied when it
    changes the image by at least ``threshold``, which is what "low contrast" looks like.
"""

import cv2
import numpy as np

ENHANCE_MODES = ("none", "clahe", "lowcontrast")

#: Separation (in [0, 1] intensity units) above which an image counts as low contrast.
LOW_CONTRAST_THRESHOLD = 0.04


def percentile_threshold(img, percentile=99.9):
    """Clamp every channel to its ``percentile`` value to cut bright outliers.

    Mirrors cellSAM: the percentile is computed over non-zero pixels only, so mostly-black
    channels are left alone instead of being blown up by the stretch.
    """
    out = np.array(img, copy=True)
    if out.ndim == 2:
        out = out[..., None]
        squeeze = True
    else:
        squeeze = False

    for chan in range(out.shape[-1]):
        current = out[..., chan]
        nonzero = current[current != 0]
        if nonzero.size == 0:
            continue
        # cast to the channel dtype so numpy does not warn about mixed-type clipping
        img_max = np.percentile(nonzero, percentile).astype(current.dtype)
        np.clip(current, None, img_max, out=current)

    return out[..., 0] if squeeze else out


def apply_clahe(img, clip_limit=2.0, tile_grid_size=8):
    """Contrast Limited Adaptive Histogram Equalisation, applied per channel."""
    image = _as_uint8(img)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size))
    )
    if image.ndim == 2:
        return clahe.apply(image)

    out = np.empty_like(image)
    for chan in range(image.shape[-1]):
        out[..., chan] = clahe.apply(image[..., chan])
    return out


def contrast_metrics(img, tile_grid_size=8):
    """``(mean_diff, mean_std)`` between the image and its CLAHE version, in [0, 1] units."""
    image = _as_uint8(img)
    gray = image.mean(axis=2) if image.ndim == 3 else image
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    equalized = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(int(tile_grid_size), int(tile_grid_size))
    ).apply(gray)
    diff = np.abs(gray.astype(np.float32) - equalized.astype(np.float32)) / 255.0
    diff = diff[diff > 0]
    if diff.size == 0:
        return 0.0, 0.0
    return float(np.median(diff)), float(np.std(diff))


def is_low_contrast(img, threshold=LOW_CONTRAST_THRESHOLD, tile_grid_size=8):
    """Whether CLAHE has to work hard on this image, i.e. it looks low contrast."""
    mean_diff, _ = contrast_metrics(img, tile_grid_size=tile_grid_size)
    return mean_diff >= threshold


def enhance_low_contrast(
    img,
    threshold=LOW_CONTRAST_THRESHOLD,
    clip_limit=2.0,
    tile_grid_size=8,
    gamma=1.0,
):
    """Percentile clipping + CLAHE (+ optional gamma), but only for low-contrast images."""
    if not is_low_contrast(img, threshold=threshold, tile_grid_size=tile_grid_size):
        return img
    out = apply_clahe(percentile_threshold(img), clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    if gamma != 1.0:
        out = _adjust_gamma(out, gamma)
    return out


def maybe_enhance(img, mode="none", **kwargs):
    """Dispatch on ``mode``; unknown modes raise instead of silently doing nothing."""
    if mode is None or mode == "none":
        return img
    if mode == "clahe":
        return apply_clahe(percentile_threshold(img), **kwargs)
    if mode == "lowcontrast":
        return enhance_low_contrast(img, **kwargs)
    raise ValueError(f"unknown enhance mode {mode!r}, expected one of {ENHANCE_MODES}")


def _as_uint8(img):
    """CLAHE and LUTs need 8-bit input; anything else is scaled and cast."""
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    scaled = img.astype(np.float32)
    low, high = float(scaled.min()), float(scaled.max())
    if high > low:
        scaled = (scaled - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _adjust_gamma(img, gamma):
    lut = ((np.arange(256) / 255.0) ** float(gamma) * 255.0).astype(np.uint8)
    return cv2.LUT(np.ascontiguousarray(img), lut)

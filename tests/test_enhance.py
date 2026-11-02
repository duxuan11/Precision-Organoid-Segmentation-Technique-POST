"""Tests for the cellSAM-inspired contrast enhancement used as TinySAM encoder input.

    python -m unittest tests.test_enhance -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from POST.utils import enhance  # noqa: E402


def _brightfield_like(shape=(64, 64), low_contrast=True, seed=0):
    """Synthetic single-object brightfield image: bright background, darker blob."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    blob = ((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2) < (min(shape) / 4) ** 2
    base = np.where(blob, 120, 150) if low_contrast else np.where(blob, 10, 240)
    img = np.clip(base + rng.normal(0, 2, shape), 0, 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)


class PercentileThresholdTest(unittest.TestCase):
    def test_clips_bright_outliers_and_leaves_other_pixels_alone(self):
        img = np.full((40, 40, 3), 100, dtype=np.uint8)
        img[0, 0] = 255  # a single hot (e.g. bubble specular) pixel

        out = enhance.percentile_threshold(img, percentile=99.0)

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLess(int(out[0, 0].max()), 255)
        np.testing.assert_array_equal(out[1:, 1:], img[1:, 1:])

    def test_blank_channel_stays_blank(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)

        out = enhance.percentile_threshold(img)

        np.testing.assert_array_equal(out, img)


class ApplyClaheTest(unittest.TestCase):
    def test_preserves_shape_and_dtype(self):
        img = _brightfield_like()

        out = enhance.apply_clahe(img, clip_limit=2.0, tile_grid_size=8)

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_increases_contrast_of_a_flat_image(self):
        img = _brightfield_like(low_contrast=True)

        out = enhance.apply_clahe(img)

        self.assertGreater(float(out.std()), float(img.std()))


class MaybeEnhanceTest(unittest.TestCase):
    def test_none_mode_returns_the_input_unchanged(self):
        img = _brightfield_like()

        np.testing.assert_array_equal(enhance.maybe_enhance(img, "none"), img)

    def test_clahe_mode_changes_the_image(self):
        img = _brightfield_like()

        self.assertFalse(np.array_equal(enhance.maybe_enhance(img, "clahe"), img))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            enhance.maybe_enhance(_brightfield_like(), "sharpen")


class EnhanceLowContrastTest(unittest.TestCase):
    def test_low_contrast_mode_is_gated_and_leaves_high_contrast_input_alone(self):
        img = _brightfield_like(low_contrast=False)  # 10 vs 240: already high contrast

        np.testing.assert_array_equal(enhance.maybe_enhance(img, "lowcontrast"), img)

    def test_low_contrast_mode_applies_percentile_then_clahe_when_the_gate_opens(self):
        img = _brightfield_like(low_contrast=True)
        expected = enhance.apply_clahe(enhance.percentile_threshold(img))

        out = enhance.enhance_low_contrast(img, threshold=-1.0, gamma=1.0)

        np.testing.assert_array_equal(out, expected)

    def test_low_contrast_mode_activates_on_a_flat_low_contrast_image(self):
        img = _brightfield_like(low_contrast=True)

        out = enhance.maybe_enhance(img, "lowcontrast")

        self.assertFalse(np.array_equal(out, img))


if __name__ == "__main__":
    unittest.main()

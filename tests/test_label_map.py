"""Tests for the instance label map produced from TinySAM masks.

Runs without the real weights (fake encoder/decoder ONNX sessions).

    python -m unittest tests.test_label_map -v
"""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from POST.tinysam import tiny_sam  # noqa: E402


def _square(size, side=10):
    """A filled square in the top-left corner, so it matches a (0, 0, side, side) bbox."""
    mask = np.zeros((size, size), dtype=bool)
    mask[0:side, 0:side] = True
    return mask


class BuildLabelMapTest(unittest.TestCase):
    def test_each_mask_gets_its_own_sequential_id(self):
        masks = [_square(64), _square(64)]
        coords = [[0, 0, 10, 10], [30, 30, 40, 40]]

        labels = tiny_sam.build_label_map(masks, coords, 64, 64)

        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(labels.shape, (64, 64))
        self.assertEqual(labels[0, 0], 1)
        self.assertEqual(labels[5, 5], 1)
        self.assertEqual(labels[35, 35], 2)
        self.assertEqual(labels[20, 20], 0)

    def test_later_mask_overwrites_earlier_on_overlap(self):
        masks = [_square(64), _square(64)]
        coords = [[0, 0, 10, 10], [0, 0, 10, 10]]

        labels = tiny_sam.build_label_map(masks, coords, 64, 64)

        self.assertEqual(labels[5, 5], 2)

    def test_masks_outside_the_canvas_are_clipped(self):
        labels = tiny_sam.build_label_map([_square(64, side=40)], [[-5, -5, 35, 35]], 64, 64)

        self.assertEqual(labels[30, 30], 1)
        self.assertEqual(labels[63, 63], 0)

    def test_empty_mask_list_gives_an_empty_map(self):
        labels = tiny_sam.build_label_map([], [], 32, 32)

        self.assertEqual(int(labels.max()), 0)

    def test_mask_process_exposes_the_label_map(self):
        result = tiny_sam.mask_process(64, 64)
        result.append_mask(_square(64))
        result.append_coord([0, 0, 10, 10])

        self.assertEqual(result.label_map()[5, 5], 1)


class SaveLabelMapTest(unittest.TestCase):
    def test_png_round_trip_is_16_bit(self):
        labels = tiny_sam.build_label_map([_square(64)], [[0, 0, 10, 10]], 64, 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "mask.png")

            tiny_sam.save_label_map(path, labels)

            reloaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        self.assertEqual(reloaded.dtype, np.uint16)
        self.assertEqual(int(reloaded[5, 5]), 1)
        self.assertEqual(int(reloaded[50, 50]), 0)

    def test_npy_round_trip_preserves_ids(self):
        labels = tiny_sam.build_label_map([_square(64), _square(64)], [[0, 0, 10, 10], [30, 30, 40, 40]], 64, 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mask.npy")

            tiny_sam.save_label_map(path, labels)

            reloaded = np.load(path)

        np.testing.assert_array_equal(reloaded, labels)

    def test_png_rejects_ids_that_do_not_fit_in_16_bit(self):
        labels = np.full((4, 4), 70000, dtype=np.uint32)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                tiny_sam.save_label_map(os.path.join(tmp, "mask.png"), labels)


class _FakeEncoderSession:
    def __init__(self):
        self.inputs = []

    def run(self, output_names, input_feed):
        self.inputs.append(np.array(input_feed["images"]))
        return [np.zeros((1, 256, 64, 64), dtype=np.float32)]


class _FakeDecoderSession:
    def run(self, output_names, input_feed):
        coords = input_feed["point_coords"]
        height, width = int(input_feed["orig_im_size"][0]), int(input_feed["orig_im_size"][1])
        masks = np.zeros((coords.shape[0], 1, height, width), dtype=np.float32)
        for index, box in enumerate(coords):
            x1 = int(round(float(box[0][0]) * width / 1024))
            y1 = int(round(float(box[0][1]) * height / 1024))
            masks[index, 0, y1:y1 + 60, x1:x1 + 60] = 1.0
        scores = np.full((coords.shape[0], 1), 0.9, dtype=np.float32)
        return [None, scores, masks]


class _PatchedSessions:
    """Route ort.InferenceSession() to the fake encoder/decoder by model path."""

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self._original = tiny_sam.ort.InferenceSession

    def __enter__(self):
        sessions = {"encoder.onnx": self.encoder, "decoder.onnx": self.decoder}

        def factory(path, providers=None, **kwargs):
            for key, session in sessions.items():
                if path.endswith(key):
                    return session
            raise AssertionError(f"unexpected model path: {path}")

        tiny_sam.ort.InferenceSession = factory
        return self

    def __exit__(self, *exc_info):
        tiny_sam.ort.InferenceSession = self._original
        return False


def _low_contrast_image(size=(900, 700)):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    blob = ((yy - size[1] / 2) ** 2 + (xx - size[0] / 2) ** 2) < (min(size) / 3) ** 2
    base = np.where(blob, 120, 148)
    img = np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(img[..., None], 3, axis=2))


class TinySamIntegrationTest(unittest.TestCase):
    def _run(self, enhance_mode):
        encoder = _FakeEncoderSession()
        with _PatchedSessions(encoder, _FakeDecoderSession()):
            tinysam = tiny_sam.TinySam(
                encoder_path="encoder.onnx",
                decoder_path="decoder.onnx",
                encoder_device="cpu",
                decoder_device="cpu",
                enhance=enhance_mode,
            )
            image = _low_contrast_image()
            tinysam.set_image(image)
            tinysam._get_image_embedding()
            masks = tinysam.predict_mask_from_bboxes(np.array([[100, 100, 300, 300]], dtype=np.int32))
        return tinysam, encoder, masks

    def test_label_map_matches_the_original_image_size(self):
        _, _, masks = self._run("none")

        labels = masks.label_map((700, 900))

        self.assertEqual(labels.shape, (700, 900))
        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(sorted(np.unique(labels).tolist()), [0, 1])

    def test_enhancement_reaches_the_encoder_input(self):
        _, plain, _ = self._run("none")
        _, enhanced, _ = self._run("clahe")

        self.assertEqual(len(plain.inputs), 1)
        self.assertEqual(len(enhanced.inputs), 1)
        self.assertEqual(plain.inputs[0].shape, enhanced.inputs[0].shape)
        self.assertFalse(np.allclose(plain.inputs[0], enhanced.inputs[0]))

    def test_no_enhancement_leaves_the_encoder_input_untouched(self):
        tinysam, encoder, _ = self._run("none")
        # 900x700 -> the encoder canvas is 1024x796, padded to 1024x1024 with zeros
        resized = np.array(
            _low_contrast_image().resize((1024, 796), Image.Resampling.BILINEAR), dtype=np.float32
        )
        expected = (resized - tinysam._mean) / tinysam._std

        np.testing.assert_allclose(
            encoder.inputs[0][0][:, :796, :].transpose(1, 2, 0), expected, rtol=1e-5, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()

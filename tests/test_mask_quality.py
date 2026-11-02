"""Tests for the mask quality gate (decoder IoU score) and min-area filter.

Runs without the real weights: a fake TinySAM decoder session returns both the mask
stack (outputs[2], like the real ONNX graph) and a per-box score tensor.

    python -m unittest tests.test_mask_quality -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from POST.tinysam import tiny_sam  # noqa: E402

EMBEDDING = np.zeros((1, 256, 64, 64), dtype=np.float32)


class _ScoredDecoderSession:
    """Fake decoder: one square mask per box, plus a per-box score tensor at outputs[1]."""

    def __init__(self, scores, side=40, with_scores=True):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.side = side
        self.with_scores = with_scores
        self.run_count = 0

    def run(self, output_names, input_feed):
        self.run_count += 1
        coords = input_feed["point_coords"]
        height, width = int(input_feed["orig_im_size"][0]), int(input_feed["orig_im_size"][1])
        batch = coords.shape[0]
        masks = np.zeros((batch, 1, height, width), dtype=np.float32)
        for index, box in enumerate(coords):
            x1 = int(round(float(box[0][0]) * width / 1024))
            y1 = int(round(float(box[0][1]) * height / 1024))
            masks[index, 0, y1:y1 + self.side, x1:x1 + self.side] = 1.0
        if not self.with_scores:
            return [None, None, masks]
        batch_scores = self.scores[:batch].reshape(-1, 1)
        return [None, batch_scores, masks]


def _boxes(count=3, width=1024, height=1024, step=200):
    return np.array([[30 + step * i, 30, 130 + step * i, 130] for i in range(count)], dtype=np.int32)


class IouGateTest(unittest.TestCase):
    def test_low_scoring_masks_are_dropped(self):
        session = _ScoredDecoderSession(scores=[0.9, 0.2, 0.8])

        masks = tiny_sam._compute_mask_from_bboxes(
            session, EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            iou_threshold=0.5,
        )

        self.assertEqual(len(masks.masks_list), 2)
        self.assertEqual(len(masks.coord_list), 2)
        # the survivors are boxes 0 and 2, so the first mask sits at the first box
        self.assertEqual(masks.coord_list[0][0], 30)
        self.assertEqual(masks.coord_list[1][0], 430)

    def test_gate_disabled_keeps_every_mask(self):
        session = _ScoredDecoderSession(scores=[0.9, 0.2, 0.8])

        masks = tiny_sam._compute_mask_from_bboxes(
            session, EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            iou_threshold=0.0,
        )

        self.assertEqual(len(masks.masks_list), 3)

    def test_decoder_without_score_output_keeps_every_mask(self):
        session = _ScoredDecoderSession(scores=[0.9, 0.2, 0.8], with_scores=False)

        masks = tiny_sam._compute_mask_from_bboxes(
            session, EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            iou_threshold=0.5,
        )

        self.assertEqual(len(masks.masks_list), 3)


class BroadcastScoreTest(unittest.TestCase):
    """Some TinySAM exports return a single (1, 1) score tensor instead of one per box."""

    class _Decoder:
        def __init__(self, score):
            self.score = np.array([[score]], dtype=np.float32)

        def run(self, output_names, input_feed):
            coords = input_feed["point_coords"]
            height, width = int(input_feed["orig_im_size"][0]), int(input_feed["orig_im_size"][1])
            masks = np.zeros((coords.shape[0], 1, height, width), dtype=np.float32)
            for index, box in enumerate(coords):
                x1 = int(round(float(box[0][0]) * width / 1024))
                y1 = int(round(float(box[0][1]) * height / 1024))
                masks[index, 0, y1:y1 + 40, x1:x1 + 40] = 1.0
            return [self.score, None, masks]

    def test_single_score_is_applied_to_every_box(self):
        masks = tiny_sam._compute_mask_from_bboxes(
            self._Decoder(0.2), EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            iou_threshold=0.5,
        )
        self.assertEqual(len(masks.masks_list), 0)

    def test_single_high_score_keeps_every_box(self):
        masks = tiny_sam._compute_mask_from_bboxes(
            self._Decoder(0.9), EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            iou_threshold=0.5,
        )
        self.assertEqual(len(masks.masks_list), 3)


class MinAreaTest(unittest.TestCase):
    def test_small_masks_are_dropped(self):
        session = _ScoredDecoderSession(scores=[0.9, 0.9, 0.9], side=3)  # 9 px each

        masks = tiny_sam._compute_mask_from_bboxes(
            session, EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            min_area=25,
        )

        self.assertEqual(len(masks.masks_list), 0)

    def test_min_area_zero_keeps_tiny_masks(self):
        session = _ScoredDecoderSession(scores=[0.9, 0.9, 0.9], side=3)

        masks = tiny_sam._compute_mask_from_bboxes(
            session, EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            min_area=0,
        )

        self.assertEqual(len(masks.masks_list), 3)

    def test_masks_and_coords_stay_aligned_after_filtering(self):
        session = _ScoredDecoderSession(scores=[0.9, 0.1, 0.9], side=3)

        masks = tiny_sam._compute_mask_from_bboxes(
            session, EMBEDDING, 50, _boxes(3), 1024, 1024, 1024, 1024,
            iou_threshold=0.5, min_area=25,
        )

        self.assertEqual(len(masks.masks_list), 0)
        self.assertEqual(len(masks.coord_list), 0)


if __name__ == "__main__":
    unittest.main()

"""Numerical equivalence / behaviour tests for the inference performance refactor.

These tests do not need the real ONNX weights: they inject a small fake onnxruntime
session, so they can run on any machine:

    python -m unittest discover -s tests -v

What is checked
---------------
* the batched YOLO preprocessing/post-processing returns exactly the same boxes as the
  old one-image-at-a-time path (for both dynamic and fixed batch ONNX graphs),
* the sliced detection pipeline produces the same predictions with batch_size=1 and >1,
* the new TinySAM decoder loop (no threads, cv2.boundingRect, optional mask_scale)
  produces the same masks/coordinates as the old reference implementation,
* `show_anns` produces the same canvas as the old full-size-mask implementation.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from POST.models.detector.yolov8onnx import Yolov8OnnxDetectionModel  # noqa: E402
from POST.tinysam import tiny_sam  # noqa: E402


class _FakeTensorInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeYoloOnnxSession:
    """Minimal onnxruntime-like session that produces deterministic detections.

    The returned boxes depend on the content of each image of the batch, so a batched
    call and a per-image call must agree exactly.
    """

    def __init__(self, num_classes=5, batch_dim=0, size=1024, boxes_per_image=3):
        self._num_classes = num_classes
        self._batch_dim = batch_dim
        self._size = size
        self._boxes_per_image = boxes_per_image
        self.batch_sizes = []  # records the batch dimension of every run() call

    def get_inputs(self):
        return [_FakeTensorInfo("images", [self._batch_dim, 3, self._size, self._size])]

    def get_outputs(self):
        return [_FakeTensorInfo("output0", [self._batch_dim, 4 + self._num_classes, 100])]

    def run(self, output_names, input_feed):
        images = input_feed["images"]
        self.batch_sizes.append(images.shape[0])
        num_anchors = 100
        out = np.zeros((images.shape[0], 4 + self._num_classes, num_anchors), dtype=np.float32)
        for batch_ind, image in enumerate(images):
            # a per-image signal so different images in a batch give different boxes
            signal = float(image.mean())
            for box_ind in range(self._boxes_per_image):
                cx = (box_ind + 1) * 100 + signal * 0.1
                cy = (box_ind + 1) * 120 + signal * 0.1
                w, h = 40, 30
                out[batch_ind, 0, box_ind] = cx
                out[batch_ind, 1, box_ind] = cy
                out[batch_ind, 2, box_ind] = w
                out[batch_ind, 3, box_ind] = h
                out[batch_ind, 4 + (box_ind % self._num_classes), box_ind] = 0.9 - 0.1 * box_ind
        return [out]


def _make_detector(session, confidence_threshold=0.45):
    detector = Yolov8OnnxDetectionModel(
        model_path="unused.onnx",
        model=session,
        device="cpu",
        confidence_threshold=confidence_threshold,
        category_mapping={"0": "bubble", "1": "small", "2": "median", "3": "large", "4": "other"},
        load_at_init=True,
        warmup=False,
    )
    return detector


class YoloOnnxBatchingTest(unittest.TestCase):
    def _run(self, session, images, shift_amounts=None, full_shapes=None):
        detector = _make_detector(session)
        from POST.models.predict import get_prediction

        result = get_prediction(
            image=images[0] if len(images) == 1 else images,
            detection_model=detector,
            shift_amount=shift_amounts[0] if shift_amounts and len(shift_amounts) == 1 else shift_amounts,
            full_shape=full_shapes[0] if full_shapes and len(full_shapes) == 1 else full_shapes,
        )
        return [
            (list(pred.bbox.to_xyxy()), float(pred.score.value), pred.category.id)
            for pred in result.object_prediction_list
        ]

    def test_dynamic_batch_matches_single_image(self):
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 255, (700, 900, 3), dtype=np.uint8) for _ in range(3)]
        shifts = [[0, 0], [120, 60], [240, 120]]
        full_shapes = [[1400, 1600]] * 3

        batched = self._run(FakeYoloOnnxSession(batch_dim=0), images, shifts, full_shapes)
        single = []
        for image, shift, full_shape in zip(images, shifts, full_shapes):
            single.extend(self._run(FakeYoloOnnxSession(batch_dim=0), [image], [shift], [full_shape]))

        self.assertEqual(len(batched), len(single))
        for (bbox_a, score_a, cls_a), (bbox_b, score_b, cls_b) in zip(batched, single):
            self.assertEqual(bbox_a, bbox_b)
            self.assertAlmostEqual(score_a, score_b, places=6)
            self.assertEqual(cls_a, cls_b)

    def test_single_run_per_batch(self):
        session = FakeYoloOnnxSession(batch_dim=0)
        rng = np.random.default_rng(1)
        images = [rng.integers(0, 255, (600, 800, 3), dtype=np.uint8) for _ in range(4)]
        self._run(session, images, [[0, 0]] * 4, [[600, 800]] * 4)
        self.assertEqual(session.batch_sizes, [4])

    def test_fixed_batch_graph_falls_back_to_single_images(self):
        session = FakeYoloOnnxSession(batch_dim=1)
        rng = np.random.default_rng(2)
        images = [rng.integers(0, 255, (600, 800, 3), dtype=np.uint8) for _ in range(2)]
        self._run(session, images, [[0, 0], [10, 10]], [[600, 800]] * 2)
        self.assertEqual(session.batch_sizes, [1, 1])

    def test_pipeline_batch_size_does_not_change_predictions(self):
        from POST.models.predict import get_sliced_prediction

        rng = np.random.default_rng(3)
        # 1500x1700 image with 512x512 slices -> 4x4 = 16 slices (plus overlap)
        image = rng.integers(0, 255, (1500, 1700, 3), dtype=np.uint8)
        results = {}
        for batch_size in (1, 4):
            detector = _make_detector(FakeYoloOnnxSession(batch_dim=0))
            prediction = get_sliced_prediction(
                image,
                detector,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.15,
                overlap_width_ratio=0.15,
                perform_standard_pred=False,
                postprocess_class_agnostic=True,
                verbose=0,
                batch_size=batch_size,
            )
            results[batch_size] = [
                (list(pred.bbox.to_xyxy()), round(float(pred.score.value), 6), pred.category.id)
                for pred in prediction.object_prediction_list
            ]
        self.assertGreater(len(results[1]), 0)
        self.assertEqual(results[1], results[4])


class _FakeDecoderSession:
    """Fake TinySAM mask decoder: draws one rectangle per box prompt."""

    def __init__(self):
        self.calls = 0
        self.last_output_shape = None
        self.last_orig_im_size = None

    def run(self, output_names, input_feed):
        self.calls += 1
        coords = input_feed["point_coords"]
        height, width = int(input_feed["orig_im_size"][0]), int(input_feed["orig_im_size"][1])
        self.last_orig_im_size = (height, width)
        masks = np.zeros((coords.shape[0], 1, height, width), dtype=np.float32)
        for index, box in enumerate(coords):
            x1, y1 = box[0]
            x2, y2 = box[1]
            x1 = int(round(float(x1) * width / 1024))
            x2 = int(round(float(x2) * width / 1024))
            y1 = int(round(float(y1) * height / 1024))
            y2 = int(round(float(y2) * height / 1024))
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width, max(x1 + 1, x2)))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height, max(y1 + 1, y2)))
            if index % 3 == 2:
                continue  # an empty mask, must be skipped by the caller
            masks[index, 0, y1:y2, x1:x2] = 1.0
        self.last_output_shape = masks.shape
        # outputs[2] is the full resolution mask stack
        return [None, None, masks]


def _reference_compute_mask_from_bboxes(
    decoder_session, image_embedding, bboxes_batch_size, bboxes,
    resized_width, resized_height, orig_width, orig_height,
):
    """The original implementation, kept as the equivalence reference."""
    from copy import deepcopy

    total_boxes = len(bboxes)
    bbox_labels = np.ones((total_boxes, 2))
    bbox_labels[:, 0] *= 2
    bbox_labels[:, 1] *= 3

    input_bboxes = np.asarray(bboxes)
    onnx_coord = np.stack(input_bboxes).reshape(-1, 2, 2)
    onnx_label = np.stack(bbox_labels).astype(np.float32)
    coords = deepcopy(onnx_coord).astype(float)
    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)
    onnx_coord = coords.astype("float32")

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    masks_list = tiny_sam.mask_process(resized_width, resized_height)
    threads = []
    import threading

    class Decoder_Thread(threading.Thread):
        def __init__(self, session, coords, labels, masks_list, image_embedding, onnx_mask_input,
                     onnx_has_mask_input, orig_height, orig_width, event):
            threading.Thread.__init__(self)
            self.session = session
            self.coords = coords
            self.labels = labels
            self.masks_list = masks_list
            self.image_embedding = image_embedding
            self.onnx_mask_input = onnx_mask_input
            self.onnx_has_mask_input = onnx_has_mask_input
            self.orig_height = orig_height
            self.orig_width = orig_width
            self.event = event

        def run(self):
            self.orig_height = 1024
            self.orig_width = 1024
            outputs = self.session.run(None, {
                "image_embeddings": self.image_embedding,
                "point_coords": self.coords,
                "point_labels": self.labels,
                "mask_input": self.onnx_mask_input,
                "has_mask_input": self.onnx_has_mask_input,
                "orig_im_size": np.array([self.orig_height, self.orig_width], dtype=np.float32),
            })
            batch_masks = np.squeeze(outputs[2] > 0, axis=1)
            for mask in batch_masks:
                mask_indices = np.argwhere(mask == 1)
                if len(mask_indices) <= 0:
                    continue
                y_min, x_min = mask_indices.min(axis=0)
                y_max, x_max = mask_indices.max(axis=0)
                x_max = min(x_max + 1, self.orig_width)
                y_max = min(y_max + 1, self.orig_height)
                y_slice = slice(y_min, y_max)
                x_slice = slice(x_min, x_max)
                self.masks_list.append_mask(mask[y_slice, x_slice])
                self.masks_list.append_coord([x_min, y_min, x_max, y_max])
            self.event.set()

    for i in range(0, total_boxes, bboxes_batch_size):
        coords = onnx_coord[i: min(i + bboxes_batch_size, total_boxes), ...]
        labels = onnx_label[i: min(i + bboxes_batch_size, total_boxes), ...]
        event = threading.Event()
        thread = Decoder_Thread(decoder_session, coords, labels, masks_list, image_embedding,
                                onnx_mask_input, onnx_has_mask_input, orig_height, orig_width, event)
        threads.append(thread)
        thread.start()
        event.wait()
    return masks_list


class TinySamDecoderTest(unittest.TestCase):
    def _boxes(self, count=11, width=900, height=700):
        rng = np.random.default_rng(4)
        xs = rng.integers(0, width - 120, size=count)
        ys = rng.integers(0, height - 120, size=count)
        boxes = np.stack([xs, ys, xs + rng.integers(40, 110, size=count), ys + rng.integers(40, 110, size=count)], axis=1)
        return boxes.astype(np.int32)

    def test_masks_and_coords_match_reference(self):
        boxes = self._boxes()
        embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)
        resized_width, resized_height = 1024, 796
        orig_width, orig_height = 1280, 996

        reference = _reference_compute_mask_from_bboxes(
            _FakeDecoderSession(), embedding, 4, boxes, resized_width, resized_height, orig_width, orig_height
        )
        optimized = tiny_sam._compute_mask_from_bboxes(
            _FakeDecoderSession(), embedding, 4, boxes, resized_width, resized_height, orig_width, orig_height
        )

        self.assertEqual(reference.coord_list, optimized.coord_list)
        self.assertEqual(len(reference.masks_list), len(optimized.masks_list))
        for mask_ref, mask_new in zip(reference.masks_list, optimized.masks_list):
            self.assertEqual(mask_ref.shape, mask_new.shape)
            np.testing.assert_array_equal(mask_ref, mask_new)

    def test_empty_bbox_list(self):
        masks = tiny_sam._compute_mask_from_bboxes(
            _FakeDecoderSession(), np.zeros((1, 256, 64, 64), dtype=np.float32), 4,
            np.empty((0, 4)), 1024, 1024, 1024, 1024,
        )
        self.assertEqual(len(masks.masks_list), 0)

    def test_mask_scale_matches_at_full_resolution(self):
        boxes = self._boxes(6)
        embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)
        full = tiny_sam._compute_mask_from_bboxes(
            _FakeDecoderSession(), embedding, 50, boxes, 1024, 1024, 1024, 1024, mask_scale=1.0
        )
        half = tiny_sam._compute_mask_from_bboxes(
            _FakeDecoderSession(), embedding, 50, boxes, 1024, 1024, 1024, 1024, mask_scale=0.5
        )
        # half resolution output must still cover the same area after upscaling
        self.assertEqual(len(full.masks_list), len(half.masks_list))
        for coord_full, coord_half in zip(full.coord_list, half.coord_list):
            self.assertLessEqual(abs(coord_full[0] - coord_half[0]), 4)
            self.assertLessEqual(abs(coord_full[1] - coord_half[1]), 4)
            self.assertLessEqual(abs(coord_full[2] - coord_half[2]), 4)
            self.assertLessEqual(abs(coord_full[3] - coord_half[3]), 4)


class ShowAnnsTest(unittest.TestCase):
    @staticmethod
    def _reference_process_mask(mask, coord, img):
        xmin, ymin, xmax, ymax = [int(v) for v in coord]
        img_h, img_w = img.shape[:2]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(img_w, xmax), min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            return
        h = min(mask.shape[0], ymax - ymin)
        w = min(mask.shape[1], xmax - xmin)
        new_mask = np.zeros((img_h, img_w), dtype=bool)
        new_mask[ymin:ymin + h, xmin:xmin + w] = mask[:h, :w]
        color_mask = np.concatenate([255 * np.random.uniform(0.15, 1, 3), [0.7]])
        img[new_mask] = color_mask
        contours, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255, 255), 2)

    def test_canvas_is_identical(self):
        rng = np.random.default_rng(5)
        masks, coords = [], []
        for _ in range(6):
            height, width = rng.integers(5, 40, size=2)
            mask = rng.random((height, width)) > 0.3
            masks.append(mask)
            coords.append([int(rng.integers(0, 60)), int(rng.integers(0, 60)),
                           int(rng.integers(60, 120)), int(rng.integers(60, 120))])

        reference_canvas = np.zeros((120, 120, 4))
        optimized_canvas = np.zeros((120, 120, 4))
        np.random.seed(0)
        for mask, coord in zip(masks, coords):
            self._reference_process_mask(mask, coord, reference_canvas)
        np.random.seed(0)
        for mask, coord in zip(masks, coords):
            tiny_sam.process_mask(mask, coord, optimized_canvas)
        np.testing.assert_array_equal(reference_canvas, optimized_canvas)


if __name__ == "__main__":
    unittest.main()


class NmsEquivalenceTest(unittest.TestCase):
    def _random_predictions(self, count=200, seed=7):
        rng = np.random.default_rng(seed)
        xs = rng.integers(0, 900, count).astype(np.float32)
        ys = rng.integers(0, 900, count).astype(np.float32)
        ws = rng.integers(20, 200, count).astype(np.float32)
        hs = rng.integers(20, 200, count).astype(np.float32)
        scores = rng.permutation(count).astype(np.float32) / count  # distinct scores
        classes = rng.integers(0, 3, count).astype(np.float32)
        return np.stack([xs, ys, xs + ws, ys + hs, scores, classes], axis=1)

    def test_vectorised_nms_matches_python_reference(self):
        import torch
        from POST.postprocess.combine import _nms_torch_reference, nms

        predictions = self._random_predictions()
        tensor = torch.from_numpy(predictions)
        for metric in ("IOU", "IOS"):
            for threshold in (0.3, 0.5, 0.7):
                new_keep = nms(tensor, match_metric=metric, match_threshold=threshold)
                ref_keep = _nms_torch_reference(tensor, match_metric=metric, match_threshold=threshold)
                self.assertEqual(sorted(int(i) for i in new_keep), sorted(int(i) for i in ref_keep),
                                 f"metric={metric} threshold={threshold}")

    def test_numpy_nms_matches_torchvision(self):
        import torch
        from torchvision.ops import nms as torchvision_nms
        from POST.utils.yolov8onnx import greedy_nms

        predictions = self._random_predictions(seed=11)
        keep_numpy = greedy_nms(predictions[:, :4], predictions[:, 4], match_threshold=0.5, match_metric="IOU")
        keep_torch = torchvision_nms(
            torch.from_numpy(predictions[:, :4]), torch.from_numpy(predictions[:, 4]), 0.5
        ).tolist()
        self.assertEqual(sorted(keep_numpy), sorted(int(i) for i in keep_torch))

    def test_inplace_shift_matches_allocating_version(self):
        from POST.models.prediction import ObjectPrediction, _shift_prediction_inplace

        for shift in ([0, 0], [120, 60], [1040, 870]):
            original = ObjectPrediction(bbox=[10, 20, 110, 220], category_id=2, category_name="median",
                                        score=np.float32(0.87), shift_amount=list(shift))
            expected = original.get_shifted_object_prediction()
            inplace = _shift_prediction_inplace(
                ObjectPrediction(bbox=[10, 20, 110, 220], category_id=2, category_name="median",
                                 score=np.float32(0.87), shift_amount=list(shift))
            )
            self.assertEqual(inplace.bbox.to_xyxy(), expected.bbox.to_xyxy())
            self.assertEqual(inplace.bbox.shift_amount, [0, 0])
            self.assertEqual(inplace.category.id, expected.category.id)
            self.assertEqual(inplace.score.value, expected.score.value)
            self.assertIsInstance(inplace.score.value, float)

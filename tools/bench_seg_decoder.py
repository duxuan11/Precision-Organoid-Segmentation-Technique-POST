"""Benchmark: TinySAM mask decoding (old threaded loop vs new batched loop).

Uses a fake decoder session that returns realistic (blobby) masks, so no weights are
needed and the timings isolate the host-side overhead that was removed.

    python tools/bench_seg_decoder.py --boxes 300
"""

import argparse
import os
import sys
import threading
import time
import tracemalloc
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from POST.tinysam import tiny_sam  # noqa: E402


class FakeDecoderSession:
    """Returns one filled ellipse-ish blob per box prompt."""

    def __init__(self):
        self.calls = 0

    def run(self, output_names, input_feed):
        self.calls += 1
        coords = input_feed["point_coords"]
        height = int(input_feed["orig_im_size"][0])
        width = int(input_feed["orig_im_size"][1])
        masks = np.zeros((coords.shape[0], 1, height, width), dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        for index, box in enumerate(coords):
            (x1, y1), (x2, y2) = box[0], box[1]
            cx, cy = float(x1 + x2) / 2 * width / 1024, float(y1 + y2) / 2 * height / 1024
            rx = max(2.0, float(x2 - x1) / 2 * width / 1024)
            ry = max(2.0, float(y2 - y1) / 2 * height / 1024)
            masks[index, 0] = (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0
        return [None, None, masks]


def reference_compute_mask_from_bboxes(session, embedding, batch_size, bboxes,
                                      resized_width, resized_height, orig_width, orig_height):
    """The original implementation (thread per batch + np.argwhere per mask)."""
    total_boxes = len(bboxes)
    bbox_labels = np.ones((total_boxes, 2))
    bbox_labels[:, 0] *= 2
    bbox_labels[:, 1] *= 3
    onnx_coord = np.stack(np.asarray(bboxes)).reshape(-1, 2, 2)
    onnx_label = np.stack(bbox_labels).astype(np.float32)
    coords = deepcopy(onnx_coord).astype(float)
    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)
    onnx_coord = coords.astype("float32")
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    masks_list = tiny_sam.mask_process(resized_width, resized_height)

    class DecoderThread(threading.Thread):
        def __init__(self, coords, labels, event):
            threading.Thread.__init__(self)
            self.coords = coords
            self.labels = labels
            self.event = event

        def run(self):
            outputs = session.run(None, {
                "image_embeddings": embedding,
                "point_coords": self.coords,
                "point_labels": self.labels,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array([1024, 1024], dtype=np.float32),
            })
            batch_masks = np.squeeze(outputs[2] > 0, axis=1)
            for mask in batch_masks:
                mask_indices = np.argwhere(mask == 1)
                if len(mask_indices) <= 0:
                    continue
                y_min, x_min = mask_indices.min(axis=0)
                y_max, x_max = mask_indices.max(axis=0)
                masks_list.append_mask(mask[y_min:y_max + 1, x_min:x_max + 1])
                masks_list.append_coord([x_min, y_min, min(x_max + 1, 1024), min(y_max + 1, 1024)])
            self.event.set()

    for i in range(0, total_boxes, batch_size):
        coords = onnx_coord[i:min(i + batch_size, total_boxes), ...]
        labels = onnx_label[i:min(i + batch_size, total_boxes), ...]
        event = threading.Event()
        thread = DecoderThread(coords, labels, event)
        thread.start()
        event.wait()
    return masks_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    boxes = np.stack([
        rng.integers(0, 800, args.boxes),
        rng.integers(0, 800, args.boxes),
    ], axis=1)
    boxes = np.concatenate([boxes, boxes + rng.integers(30, 150, (args.boxes, 2))], axis=1).astype(np.int32)
    embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)

    print(f"{args.boxes} boxes, decoder batch size {args.batch_size}")
    for label, function, kwargs in (
        ("old (thread per batch + argwhere)", reference_compute_mask_from_bboxes, {}),
        ("new (direct call + cv2.boundingRect)", tiny_sam._compute_mask_from_bboxes, {}),
        ("new, mask_scale=0.5 (less VRAM)", tiny_sam._compute_mask_from_bboxes, {"mask_scale": 0.5}),
    ):
        session = FakeDecoderSession()
        tracemalloc.start()
        start = time.time()
        if function is reference_compute_mask_from_bboxes:
            result = function(session, embedding, args.batch_size, boxes, 1024, 1024, 1024, 1024)
        else:
            result = function(session, embedding, args.batch_size, boxes, 1024, 1024, 1024, 1024, **kwargs)
        duration = time.time() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {label:38s} {duration:7.3f}s  peak host mem {peak / 1e6:7.1f} MB  masks={len(result.masks_list)}")


if __name__ == "__main__":
    main()

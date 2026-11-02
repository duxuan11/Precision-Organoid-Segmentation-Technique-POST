"""Benchmark: does batching the sliced YOLO inference actually help?

Runs the real slicing + ONNX + post-process pipeline against a synthetic yolov8-shaped
ONNX graph (so it works without the real weights) and reports timing for several batch
sizes, verifying that all of them return identical predictions.

    python tools/bench_slice_batching.py --img img/example1.tiff --batch-sizes 1 2 4 8 16
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_dummy_yolo_onnx(path, num_classes=5, num_anchors=8400, size=1024):
    """Create a yolov8-shaped ONNX graph with a dynamic batch dimension.

    The output of every image is scaled by the mean of that image, so each image of a
    batch produces different detections while the overall work stays negligible (this
    isolates the pipeline overhead that batching removes).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    channels = 4 + num_classes

    pattern = np.zeros((1, channels, num_anchors), dtype=np.float32)
    for anchor in range(num_anchors):
        pattern[0, 0, anchor] = 300 + (anchor * 37) % 700  # cx
        pattern[0, 1, anchor] = 300 + (anchor * 53) % 700  # cy
        pattern[0, 2, anchor] = 100 + anchor % 80  # w
        pattern[0, 3, anchor] = 80 + anchor % 60  # h
        for class_ind in range(num_classes):
            # ~0.6 once multiplied by 2 * mean(image) below
            pattern[0, 4 + class_ind, anchor] = 0.3 + 0.02 * ((anchor + class_ind) % 5)

    reduce_mean = helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0, axes=[1, 2, 3])
    unsqueeze = helper.make_node("Unsqueeze", ["mean", "unsqueeze_axes"], ["mean_3d"])
    scale = helper.make_node("Mul", ["mean_3d", "two"], ["weight"])
    expand = helper.make_node("Mul", ["weight", "pattern"], ["output0"])

    unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [2], [1, 2])
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two")
    pattern_tensor = numpy_helper.from_array(pattern, name="pattern")

    graph = helper.make_graph(
        [reduce_mean, unsqueeze, scale, expand],
        "dummy_yolov8",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", channels, num_anchors])],
        initializer=[unsqueeze_axes, two, pattern_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="img/example1.tiff")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a real yolov8 onnx model. When omitted a negligible dummy graph is used, "
        "which measures only the pipeline overhead that batching removes.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--slice-height", type=int, default=1024)
    parser.add_argument("--slice-width", type=int, default=1224)
    parser.add_argument("--warmup", type=int, default=1, help="number of untimed runs per configuration")
    parser.add_argument("--runs", type=int, default=2, help="number of timed runs per configuration")
    args = parser.parse_args()

    from POST.models.auto_model import AutoDetectionModel
    from POST.models.predict import get_sliced_prediction

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = args.model or build_dummy_yolo_onnx(os.path.join(tmp_dir, "dummy_yolo.onnx"))

        reference = None
        print(f"{'batch':>6} {'runs':>5} {'dets':>7} {'time/run (s)':>13} {'speedup':>8}  identical")
        baseline_time = None
        for batch_size in args.batch_sizes:
            detector = AutoDetectionModel.from_pretrained(
                model_type="yolov8onnx",
                model_path=model_path,
                confidence_threshold=0.45,
                category_mapping={"0": "bubble", "1": "small", "2": "median", "3": "large", "4": "other"},
                device="cpu",
            )
            for _ in range(args.warmup):
                get_sliced_prediction(
                    args.img, detector, slice_height=args.slice_height, slice_width=args.slice_width,
                    perform_standard_pred=False, verbose=0, batch_size=batch_size,
                )

            durations = []
            predictions = None
            for _ in range(args.runs):
                start = time.time()
                result = get_sliced_prediction(
                    args.img, detector, slice_height=args.slice_height, slice_width=args.slice_width,
                    perform_standard_pred=False, verbose=0, batch_size=batch_size,
                )
                durations.append(time.time() - start)
                predictions = [
                    (tuple(pred.bbox.to_xyxy()), round(float(pred.score.value), 6), pred.category.id)
                    for pred in result.object_prediction_list
                ]

            if reference is None:
                reference = predictions
            identical = predictions == reference
            mean_time = float(np.mean(durations))
            if baseline_time is None:
                baseline_time = mean_time
            print(
                f"{batch_size:6d} {len(durations):5d} {len(predictions):7d} {mean_time:13.3f} "
                f"{baseline_time / mean_time:7.2f}x  {identical}"
            )


if __name__ == "__main__":
    main()

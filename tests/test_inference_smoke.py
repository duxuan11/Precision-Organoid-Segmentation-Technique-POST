"""End-to-end smoke test for inference.py without the real weights.

Builds tiny synthetic ONNX graphs that mimic the shapes of the three models used by the
pipeline (yolov8 detector, TinySAM encoder, TinySAM decoder), then runs `inference.main()`
and checks that a visualization is produced.

    python -m unittest tests.test_inference_smoke -v

Skipped when `onnx` is not installed (it is only needed to synthesise the test models).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import onnx  # noqa: F401
    from onnx import TensorProto, helper, numpy_helper

    HAVE_ONNX = True
except Exception:  # pragma: no cover
    HAVE_ONNX = False


def _build_encoder(path, embedding_size=64, channels=256):
    """images (1,3,1024,1024) -> embeddings (1,256,64,64)."""
    zeros = numpy_helper.from_array(np.zeros((1, channels, embedding_size, embedding_size), dtype=np.float32), name="emb")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["emb"], ["image_embeddings"])],
        "fake_encoder",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 1024, 1024])],
        [helper.make_tensor_value_info("image_embeddings", TensorProto.FLOAT,
                                       [1, channels, embedding_size, embedding_size])],
        initializer=[zeros],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _build_decoder(path, embedding_size=64, channels=256):
    """A mask decoder that marks a filled square per box prompt (shapes match TinySAM)."""
    # batch size is taken from the point_coords tensor so the graph works for any batch
    nodes = [
        helper.make_node("Shape", ["point_coords"], ["coords_shape"]),
        helper.make_node("Gather", ["coords_shape", "zero_i"], ["batch"], axis=0),
        helper.make_node("Cast", ["batch"], ["batch_i64"], to=TensorProto.INT64),
        helper.make_node("Gather", ["orig_im_size", "zero_i"], ["out_h"], axis=0),
        helper.make_node("Cast", ["out_h"], ["out_h_i64"], to=TensorProto.INT64),
        helper.make_node("Gather", ["orig_im_size", "one_i"], ["out_w"], axis=0),
        helper.make_node("Cast", ["out_w"], ["out_w_i64"], to=TensorProto.INT64),
        helper.make_node("Unsqueeze", ["batch_i64", "axes0"], ["batch_1d"]),
        helper.make_node("Unsqueeze", ["one_i64", "axes0"], ["one_1d"]),
        helper.make_node("Unsqueeze", ["out_h_i64", "axes0"], ["h_1d"]),
        helper.make_node("Unsqueeze", ["out_w_i64", "axes0"], ["w_1d"]),
        helper.make_node("Concat", ["batch_1d", "one_1d", "h_1d", "w_1d"], ["mask_shape"], axis=0),
        helper.make_node("Expand", ["ones", "mask_shape"], ["masks"]),
        helper.make_node("Expand", ["iou", "iou_shape"], ["iou_predictions"]),
        helper.make_node("Expand", ["low_res", "low_res_shape"], ["low_res_masks"]),
    ]
    initializers = [
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero_i"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="one_i"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="one_i64"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes0"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="ones"),
        numpy_helper.from_array(np.array([[0.9]], dtype=np.float32), name="iou"),
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name="iou_shape"),
        numpy_helper.from_array(np.zeros((1, 1, 256, 256), dtype=np.float32), name="low_res"),
        numpy_helper.from_array(np.array([1, 1, 256, 256], dtype=np.int64), name="low_res_shape"),
    ]
    inputs = [
        helper.make_tensor_value_info("image_embeddings", TensorProto.FLOAT, [1, channels, embedding_size, embedding_size]),
        helper.make_tensor_value_info("point_coords", TensorProto.FLOAT, ["batch", 2, 2]),
        helper.make_tensor_value_info("point_labels", TensorProto.FLOAT, ["batch", 2]),
        helper.make_tensor_value_info("mask_input", TensorProto.FLOAT, [1, 1, 256, 256]),
        helper.make_tensor_value_info("has_mask_input", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("orig_im_size", TensorProto.FLOAT, [2]),
    ]
    outputs = [
        helper.make_tensor_value_info("iou_predictions", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("low_res_masks", TensorProto.FLOAT, [1, 1, 256, 256]),
        helper.make_tensor_value_info("masks", TensorProto.FLOAT, ["batch", 1, "h", "w"]),
    ]
    graph = helper.make_graph(nodes, "fake_decoder", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


@unittest.skipUnless(HAVE_ONNX, "onnx is required to synthesise the test models")
class InferenceSmokeTest(unittest.TestCase):
    def test_inference_end_to_end(self):
        from PIL import Image
        from tools.bench_slice_batching import build_dummy_yolo_onnx

        with tempfile.TemporaryDirectory() as tmp_dir:
            yolo_path = build_dummy_yolo_onnx(os.path.join(tmp_dir, "yolo.onnx"))
            encoder_path = _build_encoder(os.path.join(tmp_dir, "encoder.onnx"))
            decoder_path = _build_decoder(os.path.join(tmp_dir, "decoder.onnx"))
            output_path = os.path.join(tmp_dir, "result.png")

            import inference

            argv = [
                "inference.py",
                "--yolo_weight", yolo_path,
                "--seg_encoder_weight", encoder_path,
                "--seg_decoder_weight", decoder_path,
                "--img_path", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", "example1.tiff"),
                "--output_path", output_path,
                "--device", "cpu",
                "--seg_decoder_device", "cpu",
                "--seg_mask_scale", "0.5",
                "--seg_bbox_batch_size", "16",
                "--slice_height", "1024",
                "--slice_width", "1224",
                "--batch_size", "4",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                inference.main()
            finally:
                sys.argv = original_argv

            self.assertTrue(os.path.exists(output_path))
            with Image.open(output_path) as result, Image.open(argv[argv.index("--img_path") + 1]) as source:
                self.assertEqual(result.size, source.size)
                # the synthetic decoder returns full-canvas masks, so the visualization must
                # differ from the source image (i.e. masks were actually drawn)
                self.assertFalse(np.array_equal(np.asarray(result.convert("RGB")), np.asarray(source.convert("RGB"))))

    def test_cli_writes_a_16_bit_label_map(self):
        import cv2
        from PIL import Image
        from tools.bench_slice_batching import build_dummy_yolo_onnx

        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = os.path.join(tmp_dir, "masks", "label_map.png")
            argv = [
                "inference.py",
                "--yolo_weight", build_dummy_yolo_onnx(os.path.join(tmp_dir, "yolo.onnx")),
                "--seg_encoder_weight", _build_encoder(os.path.join(tmp_dir, "encoder.onnx")),
                "--seg_decoder_weight", _build_decoder(os.path.join(tmp_dir, "decoder.onnx")),
                "--img_path", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", "example1.tiff"),
                "--output_path", os.path.join(tmp_dir, "result.png"),
                "--mask_output", mask_path,
                "--enhance", "clahe",
                "--device", "cpu",
                "--seg_decoder_device", "cpu",
            ]

            import inference

            original_argv = sys.argv
            sys.argv = argv
            try:
                inference.main()
            finally:
                sys.argv = original_argv

            self.assertTrue(os.path.exists(mask_path))
            labels = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            with Image.open(argv[argv.index("--img_path") + 1]) as source:
                self.assertEqual(labels.shape, (source.size[1], source.size[0]))

        self.assertEqual(labels.dtype, np.uint16)
        self.assertGreater(int(labels.max()), 0)


if __name__ == "__main__":
    unittest.main()

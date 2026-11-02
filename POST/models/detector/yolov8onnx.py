# OBSS SAHI Tool
# Code written by Karl-Joan Alesma and Michael García, 2023.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from POST.models.detector.base import DetectionModel
from POST.models.prediction import ObjectPrediction
from POST.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from POST.utils.import_utils import check_requirements
from POST.utils.yolov8onnx import non_max_supression, xywh2xyxy


class Yolov8OnnxDetectionModel(DetectionModel):
    # tells `POST.models.predict.get_sliced_prediction` that several slices can be
    # pushed through the network in a single session.run() call
    supports_batch_inference = True

    def __init__(self, *args, iou_threshold: float = 0.7, warmup: bool = True, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max supression, defaults to 0.7.
            warmup: bool
                Run a dummy forward pass once after loading so the first real slice
                does not pay for kernel selection / memory arena growth.
        """
        self._warmup = warmup
        self._input_names = []
        self._output_names = []
        self._input_shape = (1024, 1024)  # (width, height)
        self._model_batch = 0  # 0 means the exported graph accepts a dynamic batch
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold

    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self, ort_session_kwargs: Optional[dict] = {}) -> None:
        """Detection model is initialized and set to self.model.

        Options for onnxruntime sessions can be passed as keyword arguments.
        """

        import onnxruntime

        try:

            #     EP_list = ["CPUExecutionProvider"]
            # else:
            EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            options = onnxruntime.SessionOptions()

            for key, value in ort_session_kwargs.items():
                setattr(options, key, value)

            ort_session = onnxruntime.InferenceSession(self.model_path, sess_options=options, providers=EP_list)

            self.set_model(ort_session)

        except Exception as e:
            raise TypeError("model_path is not a valid onnx model path: ", e)

    def model_warmp(self, runs: int = 1):
        """Run a dummy forward pass so the first real prediction is not slowed down by
        onnxruntime/EP initialisation. Best effort: never fails the model load."""
        if not self._warmup or self.model is None:
            return
        try:
            input_width, input_height = self._input_shape
            dummy_input = np.zeros((1, 3, input_height, input_width), dtype=np.float32)
            for _ in range(max(1, runs)):
                self.model.run(self._output_names, {self._input_names[0]: dummy_input})
        except Exception as exc:  # pragma: no cover - warmup must never break inference
            logger.warning("model warmup skipped: %s", exc)

    def _read_model_io(self) -> None:
        """Cache input/output names, the input size and the batch dimension of the graph.

        The previous implementation called `get_inputs()`/`get_outputs()` on every single
        slice and then hard coded a 1024x1024 input size.
        """
        model_inputs = self.model.get_inputs()
        model_outputs = self.model.get_outputs()
        self._input_names = [model_input.name for model_input in model_inputs]
        self._output_names = [model_output.name for model_output in model_outputs]

        shape = list(model_inputs[0].shape)  # [batch, channels, height, width]
        batch = shape[0] if len(shape) > 0 else None
        self._model_batch = batch if isinstance(batch, int) and batch > 0 else 0
        if self.image_size:
            self._input_shape = (int(self.image_size), int(self.image_size))
        else:
            input_width = shape[3] if len(shape) > 3 else None
            input_height = shape[2] if len(shape) > 2 else None
            self._input_shape = (
                input_width if isinstance(input_width, int) and input_width > 0 else 1024,
                input_height if isinstance(input_height, int) and input_height > 0 else 1024,
            )

    def set_model(self, model: Any) -> None:
        """
        Sets the underlying ONNX model.

        Args:
            model: Any
                A ONNX model
        """

        self.model = model
        self._read_model_io()
        self.model_warmp()
        # set category_mapping
        if not self.category_mapping:
            raise TypeError("Category mapping values are required")

    def _preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Resize/normalise several RGB images into one NCHW float32 batch."""
        input_width, input_height = self._input_shape
        resized = [cv2.resize(image, (input_width, input_height)) for image in images]
        batch = np.stack(resized) if len(resized) > 1 else resized[0][np.newaxis, ...]
        # uint8 -> float32 once, in place, so only a single float32 copy of the batch exists
        batch = batch.astype(np.float32)
        batch *= 1.0 / 255.0
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

    def _preprocess_image(self, image: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
        """Prepapre image for inference by resizing, normalizing and changing dimensions.

        Args:
            image: np.ndarray
                Input image with color channel order RGB.
        """
        input_image = cv2.resize(image, input_shape)

        input_image = input_image.astype(np.float32)
        input_image *= 1.0 / 255.0
        image_tensor = np.ascontiguousarray(input_image.transpose(2, 0, 1)[np.newaxis, :, :, :])

        return image_tensor

    def _post_process(
        self, outputs: np.ndarray, input_shape: Tuple[int, int], image_shape
    ) -> List[np.ndarray]:
        """Convert raw detections into per-image arrays of [x1, y1, x2, y2, score, class_id].

        `image_shape` may be a single (height, width) tuple or one such tuple per image of
        the batch.
        """
        if len(image_shape) == 2 and np.isscalar(image_shape[0]):
            image_shapes = [tuple(image_shape)]
        else:
            image_shapes = [tuple(single_shape) for single_shape in image_shape]

        input_w, input_h = input_shape
        raw_predictions = np.asarray(outputs[0])
        if raw_predictions.ndim == 2:  # single image graphs may drop the batch dim
            raw_predictions = raw_predictions[np.newaxis, ...]
        scale = np.array([input_w, input_h, input_w, input_h], dtype=np.float32)

        prediction_results = []
        for batch_ind, image_shape in enumerate(image_shapes):
            image_h, image_w = image_shape[0], image_shape[1]
            predictions = raw_predictions[batch_ind].T  # (num_anchors, 4 + num_classes)

            # Filter out object confidence scores below threshold
            scores = np.max(predictions[:, 4:], axis=1)
            keep = scores > self.confidence_threshold
            if not keep.any():
                prediction_results.append(np.empty((0, 6), dtype=np.float32))
                continue
            predictions = predictions[keep, :]
            scores = scores[keep]
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            boxes = predictions[:, :4]

            # Scale boxes to original dimensions
            boxes = np.divide(boxes, scale, dtype=np.float32)
            boxes *= np.array([image_w, image_h, image_w, image_h], dtype=np.float32)
            boxes = boxes.astype(np.int32)

            # Convert from xywh two xyxy
            boxes = xywh2xyxy(boxes).round().astype(np.int32)

            # Perform non-max supressions
            indices = non_max_supression(boxes, scores, self.iou_threshold)

            # Format the results
            prediction_result = []
            for bbox, score, label in zip(boxes[indices], scores[indices], class_ids[indices]):
                bbox = bbox.tolist()
                cls_id = int(label)
                prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])

            if prediction_result:
                prediction_results.append(np.array(prediction_result))
            else:
                prediction_results.append(np.empty((0, 6), dtype=np.float32))

        return prediction_results

    def perform_inference(self, image: Union[np.ndarray, List[np.ndarray]]):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray or list of np.ndarray
                A numpy array (or a list of them, inferred as one batch) that contains the
                image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        images = list(image) if isinstance(image, (list, tuple)) else [image]

        # A graph exported with a fixed batch dimension cannot take several images at once.
        if len(images) > 1 and self._model_batch == 1:
            predictions = []
            for single_image in images:
                self.perform_inference(single_image)
                predictions.extend(self._original_predictions)
            self._original_predictions = predictions
            return

        if self._model_batch > 1 and len(images) > self._model_batch:
            predictions = []
            for batch_start in range(0, len(images), self._model_batch):
                self.perform_inference(images[batch_start : batch_start + self._model_batch])
                predictions.extend(self._original_predictions)
            self._original_predictions = predictions
            return

        image_tensor = self._preprocess_batch(images)

        # Inference
        outputs = self.model.run(self._output_names, {self._input_names[0]: image_tensor})

        # Post-process
        prediction_results = self._post_process(
            outputs, self._input_shape, [single_image.shape[:2] for single_image in images]
        )
        self._original_predictions = prediction_results

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format:  #image_predictions_in_xyxy_format.cpu().detach().numpy()
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

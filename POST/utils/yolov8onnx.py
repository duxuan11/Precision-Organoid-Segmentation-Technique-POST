from pathlib import Path
from typing import Optional

import numpy as np

# resolved lazily by _get_torchvision_nms(): None = not initialised yet, False = unavailable
_TORCHVISION_NMS = None

#from sahi.utils.yolov8 import download_yolov8n_model


class Yolov8ONNXTestConstants:
    YOLOV8N_ONNX_MODEL_PATH = "tests/data/models/yolov8/yolov8n.onnx"


# def download_yolov8n_onnx_model(destination_path: Optional[str] = None, image_size: Optional[int] = 640):
#     if destination_path is None:
#         destination_path = Yolov8ONNXTestConstants.YOLOV8N_ONNX_MODEL_PATH
#
#     destination_path = Path(destination_path)
#     model_path = destination_path.parent / (destination_path.stem + ".pt")
#     download_yolov8n_model(model_path)
#
#     from ultralytics import YOLO
#
#     model = YOLO(model_path)
#     model.export(format="onnx")  # , imgsz=image_size)


def non_max_supression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Perform non-max supression.

    Args:
        boxes: np.ndarray
            Predicted bounding boxes, shape (num_of_boxes, 4)
        scores: np.ndarray
            Confidence for predicted bounding boxes, shape (num_of_boxes).
        iou_threshold: float
            Maximum allowed overlap between bounding boxes.

    Returns:
        np.ndarray: Filtered bounding boxes

    Notes:
        This used to be a pure python `while` loop (O(n^2) with numpy call overhead per
        iteration). It now uses `torchvision.ops.nms` when available and a vectorised
        numpy fallback otherwise.
    """
    if len(boxes) == 0:
        return []

    _nms = _get_torchvision_nms()
    if _nms is not None:
        return _nms(boxes, scores, iou_threshold)
    return _numpy_nms(boxes, scores, iou_threshold)


def _get_torchvision_nms():
    """Return a torchvision based nms function, or None when torchvision is unavailable."""
    global _TORCHVISION_NMS
    if _TORCHVISION_NMS is False:
        return None
    if _TORCHVISION_NMS is None:
        try:
            import torch
            from torchvision.ops import nms as torchvision_nms

            def _run(boxes, scores, iou_threshold):
                keep = torchvision_nms(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    torch.as_tensor(scores, dtype=torch.float32),
                    float(iou_threshold),
                )
                return keep.tolist()

            _TORCHVISION_NMS = _run
        except Exception:  # pragma: no cover - torchvision is optional
            _TORCHVISION_NMS = False
            return None
    return _TORCHVISION_NMS


def _numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """Greedy numpy NMS used when torchvision is not installed."""
    return greedy_nms(boxes, scores, match_threshold=iou_threshold, match_metric="IOU")


def greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    match_threshold: float = 0.5,
    match_metric: str = "IOU",
) -> list:
    """Vectorised greedy non-maximum suppression.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        scores: (N,) confidence scores
        match_threshold: boxes overlapping more than this are suppressed
        match_metric: "IOU" (intersection over union) or "IOS" (intersection over the
            smaller area)

    Returns:
        list of kept indices, ordered by decreasing score
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        if match_metric == "IOS":
            # normalize by the smaller of the two areas
            denominator = np.minimum(areas[current], areas[rest])
        else:
            denominator = areas[current] + areas[rest] - inter
        overlap = np.divide(inter, denominator, out=np.zeros_like(inter), where=denominator > 0)
        order = rest[overlap < match_threshold]
    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> float:
    """Compute the IOU between a selected box and other boxes.

    Args:
        box: np.ndarray
            Selected box, shape (4)
        boxes: np.ndarray
            Other boxes used for computing IOU, shape (num_of_boxes, 4).

    Returns:
        float: intersection over union
    """
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)

    Args:
        x: np.ndarray
            Input bboxes, shape (num_of_boxes, 4).

    Returns:
        np.ndarray: (num_of_boxes, 4)
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

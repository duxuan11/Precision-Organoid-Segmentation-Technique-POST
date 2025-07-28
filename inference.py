from POST.models.auto_model import AutoDetectionModel
from POST.models.predict import get_sliced_prediction
from POST.tinysam.tiny_sam import TinySam, show_anns
from PIL import Image
import numpy as np
import threading
import argparse


def read_image(img_path: str):
    image = Image.open(img_path)
    image = image.convert("RGB")
    return image


def object_detection_inference(
    image_path,
    detection_model,
    slice_height=1024,
    slice_width=1224,
    overlap_height_ratio=0.15,
    overlap_width_ratio=0.15,
):
    sahi_dec = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        perform_standard_pred=True,
        postprocess_class_agnostic=True,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    # save detection boxes and filter bubbles
    boxes = np.array(
        [
            [
                # int((box.bbox.to_xyxy()[0]+box.bbox.to_xyxy()[2])/2), # point
                # int((box.bbox.to_xyxy()[1]+box.bbox.to_xyxy()[3])/2)
                int(box.bbox.to_xyxy()[0]),
                int(box.bbox.to_xyxy()[1]),
                int(box.bbox.to_xyxy()[2]),
                int(box.bbox.to_xyxy()[3]),
            ]
            for box in sahi_dec.object_prediction_list
            if box.category.name not in ["bubble"]
        ]
    )
    return boxes


def tinysam_inference(image_path, boxes, tinysam):
    image = read_image(image_path)
    thread1 = threading.Thread(target=tinysam.set_image(image))
    thread1.start()
    mask_process = tinysam.predict_mask_from_bboxes(boxes)
    show_anns(
        mask_process.masks_list,
        mask_process.coord_list,
        mask_process.resized_width,
        mask_process.resized_height,
        image,
    )


def main():
    parser = argparse.ArgumentParser(description="test (and eval) a model")
    parser.add_argument(
        "--yolo_weight",
        default=None,
        type=str,
        required=True,
        help="test config file path.",
    )
    parser.add_argument(
        "--seg_encoder_weight",
        default=None,
        type=str,
        required=True,
        help="checkpoint file.",
    )
    parser.add_argument(
        "--seg_decoder_weight",
        default=None,
        type=str,
        required=True,
        help="checkpoint file.",
    )
    parser.add_argument("--img_path", help="The inference image path.")
    parser.add_argument(
        "--slice_height",
        default=1024,
        type=int,
        help="Height of the image slice for inference.",
    )
    parser.add_argument(
        "--slice_width",
        default=1224,
        type=int,
        help="Width of the image slice for inference.",
    )
    parser.add_argument(
        "--overlap_height_ratio",
        default=0.15,
        type=float,
        help="Overlap height ratio for image slicing.",
    )
    parser.add_argument(
        "--overlap_width_ratio",
        default=0.15,
        type=float,
        help="Overlap width ratio for image slicing.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'.",
    )
    args = parser.parse_args()
    image_path = args.img_path
    yolo_weight = args.yolo_weight
    seg_encoder_weight = args.seg_encoder_weight
    seg_decoder_weight = args.seg_decoder_weight
    device = args.device
    device = args.device if args.device else "cuda:0"
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8onnx",
        model_path=yolo_weight,
        confidence_threshold=0.45,
        category_mapping={
            "0": "bubble",
            "1": "small",
            "2": "median",
            "3": "large",
            "4": "other",
        },
        device=device,
    )
    boxes = object_detection_inference(image_path, detection_model)
    tinysam = TinySam(
        encoder_path=seg_encoder_weight,
        decoder_path=seg_decoder_weight,
    )
    tinysam_inference(image_path, boxes, tinysam)


if __name__ == "__main__":
    # Parameter from command line
    main()

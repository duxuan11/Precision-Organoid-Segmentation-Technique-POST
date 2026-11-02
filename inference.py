from POST.models.auto_model import AutoDetectionModel
from POST.models.predict import get_sliced_prediction
from POST.tinysam.tiny_sam import TinySam, save_label_map, show_anns
from POST.utils.enhance import ENHANCE_MODES
from PIL import Image
import numpy as np
import argparse


def read_image(img_path: str):
    image = Image.open(img_path)
    image = image.convert("RGB")
    return image


def object_detection_inference(
    image,
    detection_model,
    slice_height=1024,
    slice_width=1224,
    overlap_height_ratio=0.15,
    overlap_width_ratio=0.15,
    batch_size=8,
    perform_standard_pred=False,
):
    """Run the sliced YOLO detection.

    `image` can be a path or an already opened `PIL.Image` (opening the file only once
    avoids decoding the same tiff several times per run).
    """
    sahi_dec = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        perform_standard_pred=perform_standard_pred,
        postprocess_class_agnostic=True,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        batch_size=batch_size,
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


def tinysam_inference(image, boxes, tinysam, output_path=None):
    """`image` can be a path or an already opened `PIL.Image`.

    Returns the blended overlay (or None when nothing was found) and the instance label map
    at the original image resolution.
    """
    if not isinstance(image, Image.Image):
        image = read_image(image)
    # set_image() launches a background thread that computes the image embedding.
    # Wait for it explicitly before decoding. The previous code used
    # threading.Thread(target=tinysam.set_image(image)), which called set_image()
    # synchronously and passed None as the target, so the thread raised TypeError
    # and the embedding was never awaited.
    tinysam.set_image(image)
    tinysam._get_image_embedding()
    mask_process = tinysam.predict_mask_from_bboxes(boxes)
    blend_image = show_anns(
        mask_process.masks_list,
        mask_process.coord_list,
        mask_process.resized_width,
        mask_process.resized_height,
        image,
        save_path=output_path,
    )
    label_map = mask_process.label_map((image.size[1], image.size[0]))
    return blend_image, label_map


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
        "--output_path",
        default=None,
        type=str,
        help="Path to save the visualization result. If not set, the result is shown in a window.",
    )
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
        "--batch_size",
        default=8,
        type=int,
        help="Number of slices pushed through the detector in a single forward pass. "
        "Requires an onnx model exported with a dynamic batch dimension; falls back to 1 otherwise.",
    )
    parser.add_argument(
        "--standard_pred",
        action="store_true",
        help="Also run the detector once on the full image (costs roughly one extra full pass).",
    )
    parser.add_argument(
        "--seg_encoder_device",
        default="cpu",
        type=str,
        help="Device for the TinySAM image encoder, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--seg_decoder_device",
        default="cuda:0",
        type=str,
        help="Device for the TinySAM mask decoder, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--seg_bbox_batch_size",
        default=50,
        type=int,
        help="Number of boxes decoded per TinySAM decoder call. Lower it to reduce VRAM usage.",
    )
    parser.add_argument(
        "--seg_mask_scale",
        default=1.0,
        type=float,
        help="Resolution factor of the decoder output masks (1.0 = full 1024px output). "
        "Values < 1 save VRAM/memory at a small precision cost.",
    )
    parser.add_argument(
        "--seg_iou_threshold",
        default=0.5,
        type=float,
        help="Drop masks whose predicted IoU (TinySAM decoder score) is below this. 0 disables "
        "the quality gate, which is also skipped automatically when the ONNX graph "
        "returns no score tensor.",
    )
    parser.add_argument(
        "--seg_min_area",
        default=25,
        type=int,
        help="Drop masks smaller than this many pixels (0 keeps every mask).",
    )
    parser.add_argument(
        "--enhance",
        default="none",
        choices=list(ENHANCE_MODES),
        help="Contrast enhancement applied to the TinySAM encoder input only (never to the "
        "overlay). 'clahe' clips bright outliers then equalises locally, 'lowcontrast' does "
        "the same but only when the image looks low contrast. Default 'none' matches the "
        "weights as they were exported.",
    )
    parser.add_argument(
        "--mask_output",
        default=None,
        type=str,
        help="Optional path for the instance label map (0 = background, one id per mask). "
        "'.npy' keeps the uint32 ids, any other extension is written as a 16-bit image.",
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
    # read/decode the image once and hand the same object to both stages
    image = read_image(image_path)
    boxes = object_detection_inference(
        image,
        detection_model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        batch_size=args.batch_size,
        perform_standard_pred=args.standard_pred,
    )
    tinysam = TinySam(
        encoder_path=seg_encoder_weight,
        decoder_path=seg_decoder_weight,
        encoder_device=args.seg_encoder_device,
        decoder_device=args.seg_decoder_device,
        bboxes_batch_size=args.seg_bbox_batch_size,
        mask_scale=args.seg_mask_scale,
        enhance=args.enhance,
        iou_threshold=args.seg_iou_threshold,
        min_area=args.seg_min_area,
    )
    _, label_map = tinysam_inference(image, boxes, tinysam, output_path=args.output_path)
    if args.mask_output:
        save_label_map(args.mask_output, label_map)


if __name__ == "__main__":
    # Parameter from command line
    main()

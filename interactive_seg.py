from POST.tinysam.tiny_sam import TinySam, show_anns
from PIL import Image
import threading
import argparse


def read_image(img_path: str):
    image = Image.open(img_path)
    image = image.convert("RGB")
    return image


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
        "--boxes",
        default=None,
        type=list[list[int, int, int, int]],
        required=True,
        help="Bounding boxes for inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'.",
    )
    args = parser.parse_args()
    image_path = args.img_path
    seg_encoder_weight = args.seg_encoder_weight
    seg_decoder_weight = args.seg_decoder_weight
    boxes = args.boxes
    tinysam = TinySam(
        encoder_path=seg_encoder_weight,
        decoder_path=seg_decoder_weight,
    )
    tinysam_inference(image_path, boxes, tinysam)


if __name__ == "__main__":
    main()

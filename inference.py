from POST.models.auto_model import AutoDetectionModel
from POST.models.predict import get_sliced_prediction
from POST.tinysam.tiny_sam import TinySam, show_anns
from PIL import Image
import numpy as np
import threading


def read_image(img_path: str):
    image = Image.open(img_path)
    image = image.convert("RGB")
    return image


def object_detection_inference(image_path, detection_model):
    sahi_dec  = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=1024,
        slice_width=1224,
        perform_standard_pred=True,
        postprocess_class_agnostic=True,
        overlap_height_ratio=0.15,
        overlap_width_ratio=0.15,
    )
    # save detection boxes and filter bubbles
    boxes = np.array([
        [
            # int((box.bbox.to_xyxy()[0]+box.bbox.to_xyxy()[2])/2), # point
            # int((box.bbox.to_xyxy()[1]+box.bbox.to_xyxy()[3])/2)
            int(box.bbox.to_xyxy()[0]),
            int(box.bbox.to_xyxy()[1]),
            int(box.bbox.to_xyxy()[2]),
            int(box.bbox.to_xyxy()[3]),
        ]
        for box in sahi_dec.object_prediction_list if box.category.name not in ['bubble']
    ])
    return boxes


def tinysam_inference(image_path, boxes,tinysam):
    image = read_image(image_path)
    thread1 = threading.Thread(target=tinysam.set_image(image))
    thread1.start()
    mask_process = tinysam.predict_mask_from_bboxes(boxes)
    show_anns(mask_process.masks_list, mask_process.coord_list, mask_process.resized_width, mask_process.resized_height, image)


def main(image_path):
    device = "cuda:0"
    yolo_weights = "./weights/yolo_weights.onnx"
    detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8onnx",
    model_path=yolo_weights,
    confidence_threshold=0.45,
    category_mapping={"0": 'bubble', "1": 'small', "2": 'median', "3": 'large', "4": 'other'},
    device=device,
    )
    boxes = object_detection_inference(image_path,detection_model)
    tinysam = TinySam(
    encoder_path="E:\\avatarget\\deeplearning\\segmentation\\segment any organoid(SAO)\\weights\\tinysam_encoder.onnx", #E:\\avatarget\\deeplearning\\segmentation\\segment any organoid(SAO)\\weights\\tinysam_encoder.onnx
    decoder_path="E:\\avatarget\\deeplearning\\segmentation\\segment any organoid(SAO)\\weights\\tinysam_decoder.onnx")
    tinysam_inference(image_path,boxes,tinysam)


if __name__ == '__main__':
    # Parameter from command line
    image_path = "test/example1.tiff"
    main(image_path)


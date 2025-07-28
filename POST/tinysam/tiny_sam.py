import collections
import threading
from copy import deepcopy
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import concurrent.futures


class mask_process:

    def __init__(self, resized_width, resized_height):
        self.masks_list = []
        self.coord_list = []
        self.resized_width = resized_width
        self.resized_height = resized_height

    def append_mask(self, mask):
        self.masks_list.append(mask)

    def append_coord(self, coord):
        self.coord_list.append(coord)


class TinySam:
    def __init__(self, encoder_path, decoder_path):
        self._encoder_session = ort.InferenceSession(
            encoder_path, providers=["CPUExecutionProvider"]
        )  # cpu faster

        self._decoder_session = ort.InferenceSession(
            decoder_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.event = threading.Event()
        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()

        self._thread = None
        self._bboxes_batch_size = 50
        self._mean = np.array([123.675, 116.28, 103.53])
        self._std = np.array([[58.395, 57.12, 57.375]])

    # 2
    def set_name(self, name):
        self.name = name

    def _resize_image(self, image):
        self._orig_width, self._orig_height = image.size
        self.resized_width, self.resized_height = image.size
        if self._orig_width > self._orig_height:
            self.resized_width = 1024
            self.resized_height = int(1024 / self._orig_width * self._orig_height)
        else:
            self.resized_height = 1024
            self.resized_width = int(1024 / self._orig_height * self._orig_width)
        return np.array(
            image.resize(
                (self.resized_width, self.resized_height), Image.Resampling.BILINEAR
            )
        )

    # 4
    def _image_padding(self, input_array: np.ndarray):
        if self.resized_height < self.resized_width:
            input_array = np.pad(
                input_array, ((0, 0), (0, 0), (0, 1024 - self.resized_height), (0, 0))
            )
        else:
            input_array = np.pad(
                input_array, ((0, 0), (0, 0), (0, 0), (0, 1024 - self.resized_width))
            )
        return input_array

    def set_image(self, image):
        with self._lock:
            self._image = image
            self._image_embedding = self._image_embedding_cache.get(
                self._image.tobytes()
            )
        # 设置图像时，就进行embedding
        if self._image_embedding is None:
            self._thread = threading.Thread(
                target=self._compute_and_cache_image_embedding
            )
            self._thread.start()

    def _compute_and_cache_image_embedding(self):
        with self._lock:
            print("Computing image embedding...")
            image = self._resize_image(self._image)
            input_array = self._preprocess(image)
            input_array = self._image_padding(input_array)

            outputs = self._encoder_session.run(
                output_names=None,
                input_feed={"images": input_array.astype(np.float32)},
            )
            self._image_embedding = outputs[0]
            del outputs
            if len(self._image_embedding_cache) > 10:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[self._image.tobytes()] = self._image_embedding
            print("Done computing image embedding.")
            self.event.set()

    # 3
    def _preprocess(self, images: np.ndarray):
        input_tensor = (images - self._mean) / self._std
        input_tensor = input_tensor.transpose(2, 0, 1)[None]
        return input_tensor

    def _get_image_embedding(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding

    def predict_mask_from_bboxes(self, bboxes):
        with self._lock:
            self.event.wait()
            masks_list = _compute_mask_from_bboxes(
                self._decoder_session,
                self._image_embedding,
                self._bboxes_batch_size,
                bboxes,
                self.resized_width,
                self.resized_height,
                self._orig_width,
                self._orig_height,
            )
            return masks_list


class Decoder_Thread(threading.Thread):
    def __init__(
        self,
        session,
        coords,
        labels,
        masks_list,
        image_embedding,
        onnx_mask_input,
        onnx_has_mask_input,
        orig_height,
        orig_width,
        event,
    ):
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
        # 执行 ONNX Runtime 的运行操作
        self.orig_height = 1024
        self.orig_width = 1024
        outputs = self.session.run(
            None,
            {
                "image_embeddings": self.image_embedding,
                "point_coords": self.coords,
                "point_labels": self.labels,
                "mask_input": self.onnx_mask_input,
                "has_mask_input": self.onnx_has_mask_input,
                "orig_im_size": np.array(
                    [self.orig_height, self.orig_width], dtype=np.float32
                ),
            },
        )
        batch_coords = self.coords.astype(int)
        batch_masks = np.squeeze(outputs[2] > 0, axis=1)
        del outputs
        del self.session
        # mask = batch_masks[0]
        # mask = (mask > 0).astype('uint8') * 255
        # img = Image.fromarray(mask, 'L')
        # img.show()

        for mask in batch_masks:
            mask_indices = np.argwhere(mask == 1)  # Find indices where mask equals 255
            if len(mask_indices) <= 0:
                continue
            y_min, x_min = mask_indices.min(axis=0)  # Find minimum row and column
            y_max, x_max = mask_indices.max(axis=0)  # Find maximum row and column
            x_max = min(x_max + 1, self.orig_width)
            y_max = min(y_max + 1, self.orig_height)
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)
            self.masks_list.append_mask(mask[y_slice, x_slice])
            self.masks_list.append_coord([x_min, y_min, x_max, y_max])

        # 通知事件，运行操作完成
        self.event.set()


def _compute_mask_from_bboxes(
    decoder_session,
    image_embedding,
    bboxes_batch_size,
    bboxes,
    resized_width,
    resized_height,
    orig_width,
    orig_height,
):
    total_boxes = len(bboxes)
    bbox_labels = np.ones((total_boxes, 2))
    bbox_labels[:, 0] *= 2  # 第一列设置为2
    bbox_labels[:, 1] *= 3  # 第二列设置为3

    input_bboxes = np.asarray(bboxes)
    onnx_coord = np.stack(input_bboxes).reshape(-1, 2, 2)
    onnx_label = np.stack(bbox_labels).astype(np.float32)
    coords = deepcopy(onnx_coord).astype(float)
    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)
    onnx_coord = coords.astype("float32")

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # masks_list = []
    masks_list = mask_process(resized_width, resized_height)
    threads = []
    for i in range(0, total_boxes, bboxes_batch_size):
        coords = onnx_coord[i : min(i + bboxes_batch_size, total_boxes), ...]
        labels = onnx_label[i : min(i + bboxes_batch_size, total_boxes), ...]
        # 创建事件
        event = threading.Event()
        thread = Decoder_Thread(
            decoder_session,
            coords,
            labels,
            masks_list,
            image_embedding,
            onnx_mask_input,
            onnx_has_mask_input,
            orig_height,
            orig_width,
            event,
        )
        threads.append(thread)
        thread.start()
        # 等待事件
        event.wait()
        # masks, scores, _ = decoder_session.run(None, {
        #     "image_embeddings": image_embedding,
        #     "point_coords": coords,
        #     "point_labels": labels,
        #     "mask_input": onnx_mask_input,
        #     "has_mask_input": onnx_has_mask_input,
        #     "orig_im_size": np.array([orig_height, orig_width], dtype=np.float32),
        # })

    return masks_list


def process_mask(mask, coord, img):
    xmin, ymin, xmax, ymax = coord
    new_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    new_mask[ymin:ymax, xmin:xmax] = mask

    color_mask = np.concatenate(
        [255 * np.random.uniform(0.15, 1, 3), [0.7]]
    )  # np.random.random(3)

    img[new_mask] = color_mask
    contours, _ = cv2.findContours(
        new_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, (0, 0, 255, 255), 2)


def interpolate_last_two_dimensions(roi_image_nd, new_height, new_width):
    return cv2.resize(roi_image_nd, (new_width, new_height))


def show_anns(masks_list, coord_list, width, height, src_img):
    if len(masks_list) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((height, width, 4))
    # masks = masks > 0
    img[:, :, 3] = 0
    # for mask in anns:
    #     m = mask
    #     color_mask = np.concatenate([255*np.random.random(3),[0.5]])
    #     img[m] = color_mask
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_mask, mask=mask, coord=coord_list[i], img=img)
            for i, mask in enumerate(masks_list)
        ]
        concurrent.futures.wait(futures)

    mask = img[..., 3] == 0

    img[mask, 0:3] = np.array(
        src_img.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    )[mask]

    img = interpolate_last_two_dimensions(img, src_img.size[0], src_img.size[1])
    # blend_image = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(src_img.size)
    blend_image = Image.blend(
        src_img,
        Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(src_img.size),
        alpha=0.7,
    )
    blend_image.show()

import collections
import hashlib
import os
import threading
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2

from POST.utils.enhance import ENHANCE_MODES, maybe_enhance

# resolution of the square canvas the encoder/decoder operate on
MODEL_INPUT_SIZE = 1024


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

    def label_map(self, out_size=None):
        """Instance label map (0 = background, one id per mask).

        Built on the resized canvas the decoder works on, then - when ``out_size``
        ``(height, width)`` is given - scaled to the original image size with nearest
        neighbour interpolation so no new label values are invented.
        """
        labels = build_label_map(
            self.masks_list, self.coord_list, self.resized_height, self.resized_width
        )
        if out_size is None:
            return labels
        height, width = int(out_size[0]), int(out_size[1])
        if (height, width) == labels.shape:
            return labels
        # cv2.resize has no uint32 support; labels are bounded by the box count anyway
        resized = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.uint32)


def build_label_map(masks_list, coord_list, height, width):
    """Paint ``masks_list`` at ``coord_list`` into an ``(height, width)`` uint32 label map.

    Masks are painted in order, so a later mask overwrites an earlier one where they
    overlap - the same rule ``show_anns`` uses for the overlay.
    """
    labels = np.zeros((height, width), dtype=np.uint32)
    for label, (mask, coord) in enumerate(zip(masks_list, coord_list), start=1):
        xmin, ymin, xmax, ymax = [int(v) for v in coord]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        h = min(mask.shape[0], ymax - ymin)
        w = min(mask.shape[1], xmax - xmin)
        if h <= 0 or w <= 0:
            continue
        region = labels[ymin:ymin + h, xmin:xmin + w]
        region[mask[:h, :w]] = label
    return labels


def save_label_map(path, labels):
    """Write a label map to disk: ``.npy`` keeps uint32, anything else becomes a 16-bit image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.splitext(path)[1].lower() == ".npy":
        np.save(path, labels)
    else:
        max_label = int(labels.max()) if labels.size else 0
        if max_label > np.iinfo(np.uint16).max:
            raise ValueError(
                f"label map has {max_label} instances, which does not fit in a 16-bit image; "
                "save it as .npy instead"
            )
        if not cv2.imwrite(path, labels.astype(np.uint16)):
            raise IOError(f"could not write {path}")
    print(f"Label map saved to {os.path.abspath(path)}")


def _provider_list(device):
    """Map a simple device string to an onnxruntime provider list."""
    if isinstance(device, (list, tuple)):
        return list(device)
    device = (device or "cpu").lower()
    if device.startswith("cuda") or device.startswith("gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _image_cache_key(image):
    """Stable, cheap cache key that does not keep a copy of the image bytes alive."""
    return hashlib.blake2b(image.tobytes(), digest_size=16).digest()


class TinySam:
    def __init__(
        self,
        encoder_path,
        decoder_path,
        encoder_device="cpu",
        decoder_device="cuda:0",
        bboxes_batch_size=50,
        mask_scale=1.0,
        max_cache=4,
        enhance="none",
        iou_threshold=0.5,
        min_area=25,
    ):
        # The encoder runs on cpu by default: for the tiny ViT used here the host is
        # usually not the bottleneck, and keeping the embedding on the host avoids an
        # extra H2D copy per decoder batch.
        self._encoder_session = ort.InferenceSession(encoder_path, providers=_provider_list(encoder_device))
        self._decoder_session = ort.InferenceSession(decoder_path, providers=_provider_list(decoder_device))

        self.event = threading.Event()
        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()
        self._max_cache = max(1, int(max_cache))

        self._thread = None
        self._bboxes_batch_size = max(1, int(bboxes_batch_size))
        # 1.0 keeps the decoder output at full resolution (original behaviour); values
        # below 1 make the decoder emit a smaller mask and upscale the crops afterwards,
        # which cuts the (batch, 1, H, W) float output (and its GPU memory) quadratically.
        self._mask_scale = float(np.clip(mask_scale, 0.1, 1.0))
        # contrast enhancement applied to the encoder input only, never to the overlay
        if enhance not in ENHANCE_MODES:
            raise ValueError(f"enhance must be one of {ENHANCE_MODES}, got {enhance!r}")
        self._enhance = enhance
        # masks whose predicted IoU is below this are dropped (0 disables the gate)
        self._iou_threshold = float(iou_threshold)
        # masks smaller than this many pixels are dropped (0 keeps every mask)
        self._min_area = int(min_area)
        self._mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self._std = np.array([[[58.395, 57.12, 57.375]]], dtype=np.float32)

    #2
    def set_name(self, name):
        self.name = name

    def _resize_image(self, image):
        self._orig_width, self._orig_height = image.size
        self.resized_width, self.resized_height = image.size
        if self._orig_width > self._orig_height:
            self.resized_width = MODEL_INPUT_SIZE
            self.resized_height = int(MODEL_INPUT_SIZE / self._orig_width * self._orig_height)
        else:
            self.resized_height = MODEL_INPUT_SIZE
            self.resized_width = int(MODEL_INPUT_SIZE / self._orig_height * self._orig_width)
        return np.array(image.resize((self.resized_width, self.resized_height), Image.Resampling.BILINEAR))
    #4
    def _image_padding(self, input_array: np.ndarray):
        pad = MODEL_INPUT_SIZE - (self.resized_height if self.resized_height < self.resized_width else self.resized_width)
        if pad <= 0:
            return input_array
        if self.resized_height < self.resized_width:
            input_array = np.pad(input_array, ((0, 0), (0, 0), (0, pad), (0, 0)))
        else:
            input_array = np.pad(input_array, ((0, 0), (0, 0), (0, 0), (0, pad)))
        return input_array

    def set_image(self, image):
        cache_key = _image_cache_key(image)
        with self._lock:
            self._image = image
            self._cache_key = cache_key
            self._image_embedding = self._image_embedding_cache.get(cache_key)
            if self._image_embedding is not None:
                # keep recently used embeddings at the end of the LRU queue
                self._image_embedding_cache.move_to_end(cache_key)
        #设置图像时，就进行embedding
        if self._image_embedding is None:
            self._thread = threading.Thread(
                target = self._compute_and_cache_image_embedding
            )
            self._thread.start()

    def _compute_and_cache_image_embedding(self):
        # Snapshot the image under the lock and release it before the (long) inference:
        # holding the lock for the whole encoder run is what made the old
        # set_image()/_get_image_embedding() pair prone to deadlocks.
        with self._lock:
            image = self._image
            cache_key = self._cache_key
        print("Computing image embedding...")
        image_array = self._resize_image(image)
        image_array = maybe_enhance(image_array, self._enhance)
        input_array = self._preprocess(image_array)
        input_array = self._image_padding(input_array)

        outputs = self._encoder_session.run(
            output_names=None,
            input_feed={"images": np.ascontiguousarray(input_array)},
        )
        image_embedding = outputs[0]
        del outputs
        with self._lock:
            while len(self._image_embedding_cache) >= self._max_cache:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[cache_key] = image_embedding
            # only publish the embedding when it still belongs to the image that was set
            # last (a second set_image() may have happened while the encoder was running)
            if cache_key == self._cache_key:
                self._image_embedding = image_embedding
            self.event.set()
        print("Done computing image embedding.")
    #3
    def _preprocess(self, images: np.ndarray):
        # normalise in float32 instead of upcasting the whole 1024x1024x3 image to float64
        input_tensor = (images.astype(np.float32) - self._mean) / self._std
        input_tensor = input_tensor.transpose(2, 0, 1)[None]
        return input_tensor


    def _get_image_embedding(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding

    def predict_mask_from_bboxes(self, bboxes):
        # Wait for the encoder thread to finish (joins it) before running the decoder.
        # The previous version acquired self._lock and then called self.event.wait().
        # _compute_and_cache_image_embedding() holds the same lock for its whole body,
        # so if the encoder thread had not acquired the lock yet, both threads blocked
        # forever (the main thread waiting for the event, the worker waiting for the lock).
        image_embedding = self._get_image_embedding()
        return _compute_mask_from_bboxes(self._decoder_session, image_embedding, self._bboxes_batch_size, bboxes,
                            self.resized_width, self.resized_height, self._orig_width, self._orig_height,
                            mask_scale=self._mask_scale, iou_threshold=self._iou_threshold,
                            min_area=self._min_area)


def _mask_to_bbox(mask):
    """Tight (x_min, y_min, x_max, y_max) of a boolean mask, or None when the mask is empty.

    `cv2.boundingRect` on a uint8 *view* of the boolean mask is ~300x faster than the
    `np.argwhere(mask == 1)` scan that was used before (and it allocates nothing).
    """
    x, y, w, h = cv2.boundingRect(mask.view(np.uint8))
    if w == 0 or h == 0:
        return None
    return x, y, x + w, y + h


def _resize_mask(mask: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    if mask.shape[0] == new_height and mask.shape[1] == new_width:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _extract_iou_scores(outputs, batch_size):
    """Find the decoder's per-box IoU prediction tensor among the ONNX outputs.

    TinySAM exports differ: some graphs return ``[?, scores, masks]``, others only the mask
    stack. The score tensor is recognised by shape - one scalar per box prompt, values in
    [0, 1] - so a graph without it simply disables the quality gate instead of failing.
    """
    for output in outputs:
        if output is None:
            continue
        array = np.asarray(output)
        if array.ndim not in (1, 2) or array.size not in (1, batch_size):
            continue
        values = array.astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(values)) or values.min() < 0.0 or values.max() > 1.0:
            continue
        if values.size == 1 and batch_size > 1:
            # a single score for the whole batch: apply it to every box
            return np.full(batch_size, float(values[0]), dtype=np.float32)
        return values
    return None


def _compute_mask_from_bboxes(
        decoder_session, image_embedding, bboxes_batch_size, bboxes,
    resized_width, resized_height, orig_width, orig_height, mask_scale=1.0,
        iou_threshold=0.0, min_area=0):
    total_boxes = len(bboxes)
    masks_list = mask_process(resized_width, resized_height)
    if total_boxes == 0:
        return masks_list

    # boxes are in original image coordinates, the decoder expects the resized canvas
    input_bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 2, 2)
    scale = np.array([resized_width / orig_width, resized_height / orig_height], dtype=np.float32)
    onnx_coord = np.ascontiguousarray(input_bboxes * scale)
    # label 2 and 3 are the "box corner" prompt labels used by the SAM decoder
    onnx_label = np.tile(np.array([2, 3], dtype=np.float32), (total_boxes, 1))

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # the decoder upsamples the low resolution mask logits to `orig_im_size` internally
    out_height = max(1, int(round(MODEL_INPUT_SIZE * mask_scale)))
    out_width = max(1, int(round(MODEL_INPUT_SIZE * mask_scale)))
    orig_im_size = np.array([out_height, out_width], dtype=np.float32)

    mask_scale = out_height / MODEL_INPUT_SIZE

    # Run the decoder directly on the calling thread: the previous implementation started a
    # Decoder_Thread per batch and joined it immediately, so it was fully serial while still
    # paying thread creation + event signalling for every batch.
    for batch_start in range(0, total_boxes, bboxes_batch_size):
        batch_end = min(batch_start + bboxes_batch_size, total_boxes)
        outputs = decoder_session.run(None, {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord[batch_start:batch_end],
            "point_labels": onnx_label[batch_start:batch_end],
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": orig_im_size,
        })
        # (batch, 1, H, W) float32 logits -> (batch, H, W) bool without np.squeeze
        raw_masks = np.asarray(outputs[2])
        batch_masks = raw_masks[:, 0] > 0
        iou_scores = _extract_iou_scores(outputs, batch_masks.shape[0]) if iou_threshold > 0 else None
        del outputs, raw_masks

        for index, mask in enumerate(batch_masks):
            if iou_scores is not None and iou_scores[index] < iou_threshold:
                continue
            bbox = _mask_to_bbox(mask)
            if bbox is None:
                continue
            x_min, y_min, x_max, y_max = bbox
            mask_crop = mask[y_min:y_max, x_min:x_max]
            if min_area > 0 and int(mask_crop.sum()) < min_area:
                continue
            if mask_scale != 1.0:
                # map the crop computed at reduced resolution back to the 1024 canvas
                target_height = max(1, int(round(mask_crop.shape[0] / mask_scale)))
                target_width = max(1, int(round(mask_crop.shape[1] / mask_scale)))
                x_min, y_min = int(round(x_min / mask_scale)), int(round(y_min / mask_scale))
                x_max, y_max = x_min + target_width, y_min + target_height
                mask_crop = _resize_mask(mask_crop, target_height, target_width)
            masks_list.append_mask(mask_crop)
            masks_list.append_coord([x_min, y_min, x_max, y_max])

    return masks_list


def process_mask(mask, coord, img):
    xmin, ymin, xmax, ymax = [int(v) for v in coord]
    img_h, img_w = img.shape[:2]
    # Coordinates are expressed in the resized space, so for images whose aspect ratio is
    # not 1:1 the box can exceed the canvas (e.g. wide/landscape images). Clip it.
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)
    if xmax <= xmin or ymax <= ymin:
        return
    h = min(mask.shape[0], ymax - ymin)
    w = min(mask.shape[1], xmax - xmin)
    if h <= 0 or w <= 0:
        return
    mask_crop = mask[:h, :w]

    color_mask = np.concatenate([255 * np.random.uniform(0.15, 1, 3), [0.7]]) #np.random.random(3)
    # basic slicing returns a view, so this writes straight into the canvas
    img[ymin:ymin + h, xmin:xmin + w][mask_crop] = color_mask
    # Find the contours on the (small) crop and shift them to canvas coordinates instead of
    # allocating a full canvas sized mask per object.
    contours, _ = cv2.findContours(np.ascontiguousarray(mask_crop, dtype=np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour += np.array([[xmin, ymin]], dtype=contour.dtype)
    cv2.drawContours(img, contours, -1, (0, 0, 255,255), 2)

def interpolate_last_two_dimensions(roi_image_nd, new_height, new_width):
    return cv2.resize(roi_image_nd, (new_width, new_height))

def show_anns(masks_list, coord_list, width, height, src_img, save_path=None):
    """width/height 请传 resize 后的尺寸（mask_process.resized_width/resized_height）。

    save_path 为 None 时保持原有行为（弹窗预览）；指定路径则保存到磁盘，便于无界面环境批量运行。
    """
    if len(masks_list) == 0:
        return None
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((height, width, 4))
    #masks = masks > 0
    img[:,:,3] = 0
    # for mask in anns:
    #     m = mask
    #     color_mask = np.concatenate([255*np.random.random(3),[0.5]])
    #     img[m] = color_mask
    # Draw sequentially: the previous implementation used a ThreadPoolExecutor where every
    # worker wrote into the same canvas array, which is a data race.
    for mask, coord in zip(masks_list, coord_list):
        process_mask(mask, coord, img)

    mask = (img[...,3] == 0)

    img[mask,0:3] = np.array(src_img.resize((img.shape[1],img.shape[0]),Image.BILINEAR))[mask]

    img = interpolate_last_two_dimensions(img, src_img.size[0],src_img.size[1])
    #blend_image = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(src_img.size)
    blend_image = Image.blend(src_img, Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(src_img.size), alpha=0.7)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        blend_image.save(save_path)
        print(f"Result saved to {os.path.abspath(save_path)}")
    else:
        blend_image.show()
    return blend_image

import time
from typing import List, Optional
from POST.models.prediction import ObjectPrediction, PredictionResult, _shift_prediction_inplace
from POST.models.slicing import slice_image
import numpy as np
from POST.utils.cv import (
    read_image_as_pil,
)
from POST.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
    DIOUNMSPostprocess,
)
# MNS
POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
    "DIOU-NMS": DIOUNMSPostprocess,
}


def _image_to_array(image) -> np.ndarray:
    """Return an HWC array for `image` without an avoidable ndarray -> PIL -> ndarray round trip.

    Slices produced by `slice_image` are already numpy views of the source array, but the
    previous implementation converted each of them to a PIL image and back for every slice.
    """
    if isinstance(image, np.ndarray):
        return image
    return np.ascontiguousarray(read_image_as_pil(image))


def get_prediction(
    image,
    detection_model,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
    verbose: int = 0,
) -> PredictionResult:
    """
    Function for performing prediction for given image (or batch of images) using given detection_model.

    Arguments:
        image: str, np.ndarray or list/tuple of those
            Location of image, numpy image matrix to slice, or a list of images that is
            fed to the model in a single forward pass when the model supports it.
        detection_model: model.DetectionMode
        shift_amount: List or List[List]
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y] (or a list of them
            with one entry per image when a batch is given)
        full_shape: List or List[List]
            Size of the full image, should be in the form of [height, width]
        postprocess: sahi.postprocess.combine.PostprocessPredictions
        verbose: int
            0: no print (default)
            1: print prediction duration

    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    durations_in_seconds = dict()

    is_batched = isinstance(image, (list, tuple))
    images = list(image) if is_batched else [image]
    image_arrays = [_image_to_array(single_image) for single_image in images]
    shift_amounts = list(shift_amount) if is_batched else [shift_amount]
    full_shapes = list(full_shape) if is_batched else [full_shape]

    # get prediction
    time_start = time.time()
    detection_model.perform_inference(image_arrays if is_batched else image_arrays[0])
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    # process prediction
    time_start = time.time()
    detection_model.convert_original_predictions(
        shift_amount=shift_amounts if is_batched else shift_amount,
        full_shape=full_shapes if is_batched else full_shape,
    )
    object_prediction_list: List[ObjectPrediction] = [
        object_prediction
        for per_image_list in detection_model.object_prediction_list_per_image
        for object_prediction in per_image_list
    ]

    # postprocess matching predictions
    if postprocess is not None:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        image=images[0] if is_batched else image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )


def get_sliced_prediction(
    image,
    detection_model=None,
    output_file_name=None,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
    interim_dir="slices/",  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
    slice_height: int = None,
    slice_width: int = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    postprocess_type: str = "NMS",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    merge_buffer_length: int = None,
    auto_slice_resolution: bool = True,
    batch_size: int = 8,
) -> PredictionResult:

    # for profiling
    durations_in_seconds = dict()

    # create slices from full image
    time_start = time.time()
    slice_image_result = slice_image(
        image=image,
        output_file_name=output_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir if output_file_name else None,  # do not create an unused directory
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=auto_slice_resolution,
    )
    num_slices = len(slice_image_result)
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match postprocess instance
    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS.keys():
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} but given as {postprocess_type}"
        )
    elif postprocess_type == "UNIONMERGE":
        # deprecated in v0.9.3
        raise ValueError("'UNIONMERGE' postprocess_type is deprecated, use 'GREEDYNMM' instead.")
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    # create prediction input
    if verbose == 1 or verbose == 2:
        #tqdm.write(f"Performing prediction on {num_slices} number of slices.")
        print(f"Performing prediction on {num_slices} number of slices.")
    object_prediction_list = []
    # perform sliced prediction

    # Feed several slices to the detector in a single forward pass. Models that cannot
    # batch (e.g. torch models taken from ultralytics) simply get one slice at a time.
    supports_batch = bool(getattr(detection_model, "supports_batch_inference", False))
    num_batch = max(1, int(batch_size or 1)) if supports_batch else 1
    full_shape = [
        slice_image_result.original_image_height,
        slice_image_result.original_image_width,
    ]

    # access the sliced image list directly: the `images`/`starting_pixels` properties
    # rebuild a full list on every access (O(n_slices) per slice before).
    sliced_images = slice_image_result.sliced_image_list
    for group_start in range(0, num_slices, num_batch):
        group = sliced_images[group_start : group_start + num_batch]
        image_list = [sliced_image.image for sliced_image in group]
        shift_amount_list = [sliced_image.starting_pixel for sliced_image in group]

        # perform batch prediction
        if len(group) == 1:
            prediction_result = get_prediction(
                image=image_list[0],
                detection_model=detection_model,
                shift_amount=shift_amount_list[0],
                full_shape=full_shape,
            )
            batch_object_predictions = prediction_result.object_prediction_list
        else:
            prediction_result = get_prediction(
                image=image_list,
                detection_model=detection_model,
                shift_amount=shift_amount_list,
                full_shape=[full_shape] * len(group),
            )
            batch_object_predictions = prediction_result.object_prediction_list

        # convert sliced predictions to full predictions
        for object_prediction in batch_object_predictions:
            if object_prediction:  # if not empty
                object_prediction_list.append(_shift_prediction_inplace(object_prediction))

        # merge matching predictions during sliced prediction
        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            object_prediction_list = postprocess(object_prediction_list)
    del sliced_images
    #print("推理总耗时：{}秒".format(end - start))

    # perform standard prediction
    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list.extend(prediction_result.object_prediction_list)

    # merge matching predictions
    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
    )

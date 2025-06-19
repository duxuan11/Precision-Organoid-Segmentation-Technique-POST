import time
from typing import List, Optional
from POST.models.prediction import ObjectPrediction, PredictionResult
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


def get_prediction(
    image,
    detection_model,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
    verbose: int = 0,
) -> PredictionResult:
    """
    Function for performing prediction for given image using given detection_model.

    Arguments:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
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

    # read image as pil
    image_as_pil = read_image_as_pil(image)
    # get prediction
    time_start = time.time()
    detection_model.perform_inference(np.ascontiguousarray(image_as_pil))
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    # process prediction
    time_start = time.time()
    # works only with 1 batch
    detection_model.convert_original_predictions(
        shift_amount=shift_amount,
        full_shape=full_shape,
    )
    object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list

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
        image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
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
) -> PredictionResult:

    # for profiling
    durations_in_seconds = dict()

    # currently only 1 batch supported
    num_batch = 1

    # create slices from full image
    time_start = time.time()
    slice_image_result = slice_image(
        image=image,
        output_file_name=output_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
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
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        #tqdm.write(f"Performing prediction on {num_slices} number of slices.")
        print(f"Performing prediction on {num_slices} number of slices.")
    object_prediction_list = []
    # perform sliced prediction
    start = time.time()

    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
        # perform batch prediction
        one_time = time.time()
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        one_time_2 = time.time()
        #print("单次耗时：",one_time_2 - one_time)
        # convert sliced predictions to full predictions
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:  # if not empty
                object_prediction_list.append(object_prediction.get_shifted_object_prediction())

        # merge matching predictions during sliced prediction
        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            object_prediction_list = postprocess(object_prediction_list)
    end = time.time()
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

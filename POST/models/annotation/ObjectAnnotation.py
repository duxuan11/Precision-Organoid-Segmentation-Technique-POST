import copy
from typing import Dict, List, Optional
import numpy as np


class BoundingBox:
    """
    Bounding box of the annotation.
    """

    def __init__(self, box: List[float], shift_amount: List[int] = [0, 0]):
        """
        Args:
            box: List[float]
                [minx, miny, maxx, maxy]
            shift_amount: List[int]
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
        """
        if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
            raise Exception("Box coords [minx, miny, maxx, maxy] cannot be negative")
        self.minx = box[0]
        self.miny = box[1]
        self.maxx = box[2]
        self.maxy = box[3]

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

    @property
    def shift_amount(self):
        """
        Returns the shift amount of the bbox slice as [shift_x, shift_y]
        """
        return [self.shift_x, self.shift_y]

    @property
    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def get_expanded_box(self, ratio=0.1, max_x=None, max_y=None):
        w = self.maxx - self.minx
        h = self.maxy - self.miny
        y_mar = int(w * ratio)
        x_mar = int(h * ratio)
        maxx = min(max_x, self.maxx + x_mar) if max_x else self.maxx + x_mar
        minx = max(0, self.minx - x_mar)
        maxy = min(max_y, self.maxy + y_mar) if max_y else self.maxy + y_mar
        miny = max(0, self.miny - y_mar)
        box = [minx, miny, maxx, maxy]
        return BoundingBox(box)

    def to_xywh(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_coco_bbox(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return self.to_xywh()

    def to_xyxy(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_voc_bbox(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return self.to_xyxy()

    def get_shifted_box(self):
        """
        Returns: shifted BoundingBox
        """
        box = [
            self.minx + self.shift_x,
            self.miny + self.shift_y,
            self.maxx + self.shift_x,
            self.maxy + self.shift_y,
        ]
        return BoundingBox(box)

    def __repr__(self):
        return f"BoundingBox: <{(self.minx, self.miny, self.maxx, self.maxy)}, w: {self.maxx - self.minx}, h: {self.maxy - self.miny}>"


class Category:
    """
    Category of the annotation.
    """

    def __init__(self, id=None, name=None):
        """
        Args:
            id: int
                ID of the object category
            name: str
                Name of the object category
        """
        if not isinstance(id, int):
            raise TypeError("id should be integer")
        if not isinstance(name, str):
            raise TypeError("name should be string")
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"


class ObjectAnnotation:
    """
    All about an annotation such as Mask, Category, BoundingBox.
    """

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from bool_mask (2D np.ndarray)

        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )



    @classmethod
    def from_coco_bbox(
        cls,
        bbox: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from coco bbox [minx, miny, width, height]

        Args:
            bbox: List
                [minx, miny, width, height]
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        return cls(
            category_id=category_id,
            bbox=bbox,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_annotation_dict(
        cls,
        annotation_dict: Dict,
        full_shape: List[int],
        category_name: str = None,
        shift_amount: Optional[List[int]] = [0, 0],
    ):
        """
        Creates ObjectAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
            category_name: str
                Category name of the annotation
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        if annotation_dict["segmentation"]:
            return cls.from_coco_segmentation(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
        else:
            return cls.from_coco_bbox(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )

    @classmethod
    def from_imantics_annotation(
        cls,
        annotation,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from imantics.annotation.Annotation

        Args:
            annotation: imantics.annotation.Annotation
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
        """
        return cls(
            category_id=annotation.category.id,
            bool_mask=annotation.mask.array,
            category_name=annotation.category.name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        bool_mask: Optional[np.ndarray] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Args:
            bbox: List
                [minx, miny, maxx, maxy]
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        if not isinstance(category_id, int):
            raise ValueError("category_id must be an integer")
        if (bbox is None) and (bool_mask is None):
            raise ValueError("you must provide a bbox or bool_mask")

        if bool_mask is not None:
            self.mask = Mask(
                bool_mask=bool_mask,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            bbox_from_bool_mask = get_bbox_from_bool_mask(bool_mask)
            # https://github.com/obss/sahi/issues/235
            if bbox_from_bool_mask is not None:
                bbox = bbox_from_bool_mask
            else:
                raise ValueError("Invalid boolean mask.")
        else:
            self.mask = None

        # if bbox is a numpy object, convert it to python List[float]
        if type(bbox).__module__ == "numpy":
            bbox = copy.deepcopy(bbox).tolist()

        # make sure bbox coords lie inside [0, image_size]
        xmin = max(bbox[0], 0)
        ymin = max(bbox[1], 0)
        if full_shape:
            xmax = min(bbox[2], full_shape[1])
            ymax = min(bbox[3], full_shape[0])
        else:
            xmax = bbox[2]
            ymax = bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        # set bbox
        self.bbox = BoundingBox(bbox, shift_amount)

        category_name = category_name if category_name else str(category_id)
        self.category = Category(
            id=category_id,
            name=category_name,
        )

        self.merged = None

    # def to_coco_annotation(self):
    #     """
    #     Returns sahi.utils.coco.CocoAnnotation representation of ObjectAnnotation.
    #     """
    #     if self.mask:
    #         coco_annotation = CocoAnnotation.from_coco_segmentation(
    #             segmentation=self.mask.to_coco_segmentation(),
    #             category_id=self.category.id,
    #             category_name=self.category.name,
    #         )
    #     else:
    #         coco_annotation = CocoAnnotation.from_coco_bbox(
    #             bbox=self.bbox.to_xywh(),
    #             category_id=self.category.id,
    #             category_name=self.category.name,
    #         )
    #     return coco_annotation
    #
    # def to_coco_prediction(self):
    #     """
    #     Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.
    #     """
    #     if self.mask:
    #         coco_prediction = CocoPrediction.from_coco_segmentation(
    #             segmentation=self.mask.to_coco_segmentation(),
    #             category_id=self.category.id,
    #             category_name=self.category.name,
    #             score=1,
    #         )
    #     else:
    #         coco_prediction = CocoPrediction.from_coco_bbox(
    #             bbox=self.bbox.to_xywh(),
    #             category_id=self.category.id,
    #             category_name=self.category.name,
    #             score=1,
    #         )
    #     return coco_prediction
    #
    # def to_shapely_annotation(self):
    #     """
    #     Returns sahi.utils.shapely.ShapelyAnnotation representation of ObjectAnnotation.
    #     """
    #     if self.mask:
    #         shapely_annotation = ShapelyAnnotation.from_coco_segmentation(
    #             segmentation=self.mask.to_coco_segmentation(),
    #         )
    #     else:
    #         shapely_annotation = ShapelyAnnotation.from_coco_bbox(
    #             bbox=self.bbox.to_xywh(),
    #         )
    #     return shapely_annotation
    #
    # def to_imantics_annotation(self):
    #     """
    #     Returns imantics.annotation.Annotation representation of ObjectAnnotation.
    #     """
    #     try:
    #         import imantics
    #     except ImportError:
    #         raise ImportError(
    #             'Please run "pip install -U imantics" ' "to install imantics first for imantics conversion."
    #         )
    #
    #     imantics_category = imantics.Category(id=self.category.id, name=self.category.name)
    #     if self.mask is not None:
    #         imantics_mask = imantics.Mask.create(self.mask.bool_mask)
    #         imantics_annotation = imantics.annotation.Annotation.from_mask(
    #             mask=imantics_mask, category=imantics_category
    #         )
    #     else:
    #         imantics_bbox = imantics.BBox.create(self.bbox.to_xyxy())
    #         imantics_annotation = imantics.annotation.Annotation.from_bbox(
    #             bbox=imantics_bbox, category=imantics_category
    #         )
    #     return imantics_annotation

    def deepcopy(self):
        """
        Returns: deepcopy of current ObjectAnnotation instance
        """
        return copy.deepcopy(self)

    @classmethod
    def get_empty_mask(cls):
        return Mask(bool_mask=None)

    def get_shifted_object_annotation(self):
        if self.mask:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=self.mask.get_shifted_mask().full_shape,
            )
        else:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def __repr__(self):
        return f"""ObjectAnnotation<
    bbox: {self.bbox},
    mask: {self.mask},
    category: {self.category}>"""
from __future__ import division

import numpy as np

from ..abstract import Processor, Box2D
from ..backend.boxes import match
from ..backend.boxes import encode
from ..backend.boxes import decode
from ..backend.boxes import offset
from ..backend.boxes import clip
from ..backend.boxes import nms_per_class
from ..backend.boxes import merge_nms_box_with_class
from ..backend.boxes import denormalize_box
from ..backend.boxes import make_box_square
from ..backend.boxes import filter_boxes
from ..backend.boxes import scale_box


class SquareBoxes2D(Processor):
    """Transforms bounding rectangular boxes into square bounding boxes.
    """
    def __init__(self):
        super(SquareBoxes2D, self).__init__()

    def call(self, boxes2D):
        for box2D in boxes2D:
            box2D.coordinates = make_box_square(box2D.coordinates)
        return boxes2D


class DenormalizeBoxes2D(Processor):
    """Denormalizes boxes shapes to be in accordance to the original
    image size.

    # Arguments:
        image_size: List containing height and width of an image.
    """
    def __init__(self):
        super(DenormalizeBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        shape = image.shape[:2]
        for box2D in boxes2D:
            box2D.coordinates = denormalize_box(box2D.coordinates, shape)
        return boxes2D


class RoundBoxes2D(Processor):
    """Round to integer box coordinates.
    """
    def __init__(self):
        super(RoundBoxes2D, self).__init__()

    def call(self, boxes2D):
        for box2D in boxes2D:
            box2D.coordinates = [int(x) for x in box2D.coordinates]
        return boxes2D


class FilterClassBoxes2D(Processor):
    """Filters boxes with valid class names.

    # Arguments
        valid_class_names: List of strings indicating class names to be kept.
    """
    def __init__(self, valid_class_names):
        self.valid_class_names = valid_class_names
        super(FilterClassBoxes2D, self).__init__()

    def call(self, boxes2D):
        filtered_boxes2D = []
        for box2D in boxes2D:
            if box2D.class_name in self.valid_class_names:
                filtered_boxes2D.append(box2D)
        return filtered_boxes2D


class CropBoxes2D(Processor):
    """Creates a list of images cropped from the bounding boxes.

    # Arguments
        offset_scales: List of floats having x and y scales respectively.
    """
    def __init__(self):
        super(CropBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        image_crops = []
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            image_crops.append(image[y_min:y_max, x_min:x_max])
        return image_crops


class ClipBoxes2D(Processor):
    """Clips boxes coordinates into the image dimensions"""
    def __init__(self):
        super(ClipBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        image_height, image_width = image.shape[:2]
        for box2D in boxes2D:
            box2D.coordinates = clip(box2D.coordinates, image.shape[:2])
        return boxes2D


class OffsetBoxes2D(Processor):
    """Offsets the height and widht of a list of ``Boxes2D``.

    # Arguments
        offsets: Float between [0, 1].
    """
    def __init__(self, offsets):
        super(OffsetBoxes2D, self).__init__()
        self.offsets = offsets

    def call(self, boxes2D):
        for box2D in boxes2D:
            box2D.coordinates = offset(box2D.coordinates, self.offsets)
        return boxes2D


class ToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages.

    # Arguments
        class_names: List of class names ordered with respect to the
            class indices from the dataset ``boxes``.
        one_hot_encoded: Bool, indicating if scores are one hot vectors.
        default_score: Float, score to set.
        default_class: Str, class to set.
        box_method: Int, method to convert boxes to ``Boxes2D``.

    # Properties
        one_hot_encoded: Bool.
        box_processor: Callable.

    # Methods
        call()
    """
    def __init__(
            self, class_names=None, one_hot_encoded=False,
            default_score=1.0, default_class=None, box_method=0):
        if class_names is not None:
            arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        method_to_processor = {
            0: BoxesWithOneHotVectorsToBoxes2D(arg_to_class),
            1: BoxesToBoxes2D(default_score, default_class),
            2: BoxesWithClassArgToBoxes2D(arg_to_class, default_score)}
        self.box_processor = method_to_processor[box_method]
        super(ToBoxes2D, self).__init__()

    def call(self, box_data):
        return self.box_processor(box_data)


class BoxesToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages given no
    class names and score.

    # Arguments
        default_score: Float, score to set.
        default_class: Str, class to set.

    # Properties
        default_score: Float.
        default_class: Str.

    # Methods
        call()
    """
    def __init__(self, default_score=1.0, default_class=None):
        self.default_score = default_score
        self.default_class = default_class
        super(BoxesToBoxes2D, self).__init__()

    def call(self, box_data):
        boxes2D = []
        for box in box_data:
            boxes2D.append(
                Box2D(box[:4], self.default_score, self.default_class))
        return boxes2D


class BoxesWithOneHotVectorsToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages given boxes
    with scores as one hot vectors.

    # Arguments
        arg_to_class: List, of classes.

    # Properties
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class):
        self.arg_to_class = arg_to_class
        super(BoxesWithOneHotVectorsToBoxes2D, self).__init__()

    def call(self, box_data):
        boxes2D = []
        for box in box_data:
            class_scores = box[4:]
            class_arg = np.argmax(class_scores)
            score = class_scores[class_arg]
            class_name = self.arg_to_class[class_arg]
            boxes2D.append(Box2D(box[:4], score, class_name))
        return boxes2D


class BoxesWithClassArgToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages given boxes
    with class argument.

    # Arguments
        default_score: Float, score to set.
        arg_to_class: List, of classes.

    # Properties
        default_score: Float.
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class, default_score=1.0):
        self.default_score = default_score
        self.arg_to_class = arg_to_class
        super(BoxesWithClassArgToBoxes2D, self).__init__()

    def call(self, box_data):
        boxes2D = []
        for box in box_data:
            class_name = self.arg_to_class[box[-1]]
            boxes2D.append(Box2D(box[:4], self.default_score, class_name))
        return boxes2D


class RoundBoxes(Processor):
    """Rounds the floating value coordinates of the box coordinates
    into integer type.

    # Methods
        call()
    """
    def __init__(self):
        super(RoundBoxes, self).__init__()

    def call(self, boxes2D):
        for box2D in boxes2D:
            box2D.coordinates = box2D.coordinates.astype(int)
        return boxes2D


class MatchBoxes(Processor):
    """Match prior boxes with ground truth boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        iou: Float in [0, 1]. Intersection over union in which prior boxes
            will be considered positive. A positive box is box with a class
            different than `background`.
        variance: List of two floats.
    """
    def __init__(self, prior_boxes, iou=.5):
        self.prior_boxes = prior_boxes
        self.iou = iou
        super(MatchBoxes, self).__init__()

    def call(self, boxes):
        boxes = match(boxes, self.prior_boxes, self.iou)
        return boxes


class EncodeBoxes(Processor):
    """Encodes bounding boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        variances: List of two float values.
    """
    def __init__(self, prior_boxes, variances=[0.1, 0.1, 0.2, 0.2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(EncodeBoxes, self).__init__()

    def call(self, boxes):
        encoded_boxes = encode(boxes, self.prior_boxes, self.variances)
        return encoded_boxes


class DecodeBoxes(Processor):
    """Decodes bounding boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        variances: List of two float values.
    """
    def __init__(self, prior_boxes, variances=[0.1, 0.1, 0.2, 0.2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(DecodeBoxes, self).__init__()

    def call(self, boxes):
        decoded_boxes = decode(boxes, self.prior_boxes, self.variances)
        return decoded_boxes


class NonMaximumSuppressionPerClass(Processor):
    """Applies non maximum suppression per class.

    # Arguments
        nms_thresh: Float between [0, 1].
        epsilon: Float between [0, 1].
    """
    def __init__(self, nms_thresh=.45, epsilon=0.01):
        self.nms_thresh = nms_thresh
        self.epsilon = epsilon
        super(NonMaximumSuppressionPerClass, self).__init__()

    def call(self, box_data):
        box_data, class_labels = nms_per_class(
            box_data, self.nms_thresh, self.epsilon)
        return box_data, class_labels


class MergeNMSBoxWithClass(Processor):
    """Merges box coordinates with their corresponding class
    defined by `class_labels` which is decided by best box geometry
    by non maximum suppression (and not by the best scoring class)
    into a single output.
    """
    def __init__(self):
        super(MergeNMSBoxWithClass, self).__init__()

    def call(self, box_data, class_labels):
        box_data = merge_nms_box_with_class(box_data, class_labels)
        return box_data


class FilterBoxes(Processor):
    """Filters boxes outputted from function ``detect`` as
    ``Box2D`` messages.

    # Arguments
        class_names: List of class names.
        conf_thresh: Float between [0, 1].
    """
    def __init__(self, class_names, conf_thresh=0.5):
        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.arg_to_class = dict(zip(
            list(range(len(self.class_names))), self.class_names))
        super(FilterBoxes, self).__init__()

    def call(self, box_data):
        box_data = filter_boxes(box_data, self.conf_thresh)
        return box_data


class CropImage(Processor):
    """Crop images using a list of ``box2D``.
    """
    def __init__(self):
        super(CropImage, self).__init__()

    def call(self, image, box2D):
        x_min, y_min, x_max, y_max = box2D.coordinates
        return image[y_min:y_max, x_min:x_max]


class RemoveClass(Processor):
    """Remove a particular class from the pipeline.

    # Arguments
        class_names: List, indicating given class names.
        class_arg: Int, index of the class to be removed.
        renormalize: Bool, if true scores are renormalized.

    # Properties
        class_arg: Int.
        renormalize: Bool

    # Methods
        call()
    """
    def __init__(self, class_names, class_arg=None, renormalize=False):
        self.class_arg = class_arg
        self.renormalize = renormalize
        if class_arg is not None:
            del class_names[class_arg]
        super(RemoveClass, self).__init__()

    def call(self, box_data):
        if not self.renormalize and self.class_arg is not None:
            box_data = np.delete(box_data, 4 + self.class_arg, axis=1)
        elif self.renormalize:
            raise NotImplementedError
        return box_data


class ScaleBox(Processor):
    """Scale box coordinates of the prediction.

    # Arguments
        scales: Array of shape `()`, value to scale boxes.

    # Properties
        scales: Int.

    # Methods
        call()
    """
    def __init__(self):
        super(ScaleBox, self).__init__()

    def call(self, boxes, scales):
        boxes = scale_box(boxes, scales)
        return boxes

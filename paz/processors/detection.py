import numpy as np

from ..core import Processor, Box2D, ops
from .image import BGR_IMAGENET_MEAN


class SquareBoxes2D(Processor):
    """Transforms bounding rectangular boxes into square bounding boxes.
    # Arguments
        offset_scale: Float. Bounding box offset scale i.e.
            If the square bounding box has shape LxL
            the offset modifies both sizes L to be:
            L_new = L + (offset_scale * L)
    """
    def __init__(self, offset_scale=0.0):
        self.offset_scale = offset_scale
        super(SquareBoxes2D, self).__init__()

    def call(self, kwargs):
        for box_arg in range(len(kwargs['boxes2D'])):
            coordinates = kwargs['boxes2D'][box_arg].coordinates
            coordinates = ops.make_box_square(coordinates, self.offset_scale)
            kwargs['boxes2D'][box_arg].coordinates = coordinates
        return kwargs


class DenormalizeBoxes2D(Processor):
    """Denormalizes boxes shapes to be in accordance to the original image size.
    """
    def __init__(self):
        super(DenormalizeBoxes2D, self).__init__()

    def call(self, kwargs):
        image_size = kwargs['image'].shape[:2]
        for box2D in kwargs['boxes2D']:
            box2D.coordinates = ops.denormalize_box(
                box2D.coordinates, image_size)
        return kwargs


class RoundBoxes2D(Processor):
    """ Round boxes coordinates.
    """
    def __init__(self):
        super(RoundBoxes2D, self).__init__()

    def call(self, kwargs):
        boxes2D = []
        for box2D in kwargs['boxes2D']:
            box2D.coordinates = [int(x) for x in box2D.coordinates]
            boxes2D.append(box2D)
        kwargs['boxes2D'] = boxes2D
        return kwargs


class ClipBoxes2D(Processor):
    """Clips boxes coordinates into the image dimensions"""
    def __init__(self):
        super(ClipBoxes2D, self).__init__()

    def call(self, kwargs):
        image_height, image_width = kwargs['image'].shape[:2]
        for box_arg in range(len(kwargs['boxes2D'])):
            box2D = kwargs['boxes2D'][box_arg]
            x_min, y_min, x_max, y_max = box2D.coordinates
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > image_width:
                x_max = image_width
            if y_max > image_height:
                y_max = image_height
            coordinates = (x_min, y_min, x_max, y_max)
            kwargs['boxes2D'][box_arg].coordinates = coordinates
        return kwargs


class FilterClassBoxes2D(Processor):
    """Filters boxes with valid class names.
    # Arguments
        valid_class_names: List of strings indicating class names to be kept.
    """
    def __init__(self, valid_class_names):
        self.valid_class_names = valid_class_names
        super(FilterClassBoxes2D, self).__init__()

    def call(self, kwargs):
        filtered_boxes2D, boxes2D = [], kwargs['boxes2D']
        for box2D in boxes2D:
            if box2D.class_name in self.valid_class_names:
                filtered_boxes2D.append(box2D)
        kwargs['boxes2D'] = filtered_boxes2D
        return kwargs


class CropBoxes2D(Processor):
    """Creates a list of images cropped from the bounding boxes.
    # Arguments
        offset_scales: List of floats having x and y scales respectively.
        topic: String indicating the new key in the data dictionary that
            will contain the list of cropped images.
    """
    def __init__(self, offset_scales, topic='image_crops'):
        self.offset_scales = offset_scales
        self.topic = topic
        super(CropBoxes2D, self).__init__()

    def call(self, kwargs):
        image, image_crops = kwargs['image'], []
        for box2D in kwargs['boxes2D']:
            coordinates = box2D.coordinates
            coordinates = ops.apply_offsets(coordinates, self.offset_scales)
            x_min, y_min, x_max, y_max = coordinates
            image_crops.append(image[y_min:y_max, x_min:x_max])
        kwargs[self.topic] = image_crops
        return kwargs


class ToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages.
    # Arguments
        class_names: List of class names ordered with respect to the
            class indices from the dataset `boxes`.
    """
    def __init__(self, class_names=None, one_hot_encoded=False):
        if class_names is not None:
            self.arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        super(ToBoxes2D, self).__init__()

    def call(self, kwargs):
        numpy_boxes2D, boxes2D = kwargs['boxes'], []
        for numpy_box2D in numpy_boxes2D:
            if self.one_hot_encoded:
                class_name = self.arg_to_class[np.argmax(numpy_box2D[4:])]
            elif numpy_box2D.shape[-1] == 5:
                class_name = self.arg_to_class[numpy_box2D[-1]]
            elif numpy_box2D.shape[-1] == 4:
                class_name = None
            boxes2D.append(Box2D(numpy_box2D[:4], 1.0, class_name))

        # check if there are already boxes inside the `Boxes2D` topic
        if 'boxes2D' in kwargs:
            kwargs['boxes2D'].extend(boxes2D)
        else:
            kwargs['boxes2D'] = boxes2D
        return kwargs


class MatchBoxes(Processor):
    """Match prior boxes with ground truth boxes.
    #Arguments
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

    def call(self, kwargs):
        boxes = ops.match(kwargs['boxes'], self.prior_boxes, self.iou)
        kwargs['boxes'] = boxes
        return kwargs


class EncodeBoxes(Processor):
    """TODO: Encodes bounding boxes.
    """
    def __init__(self, prior_boxes, variances=[.1, .2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(EncodeBoxes, self).__init__()

    def call(self, kwargs):
        boxes = kwargs['boxes']
        encoded_boxes = ops.encode(boxes, self.prior_boxes, self.variances)
        kwargs['boxes'] = encoded_boxes
        return kwargs


class DecodeBoxes(Processor):
    """TODO: Decodes boxes.
    """
    def __init__(self, prior_boxes, variances=[.1, .2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(DecodeBoxes, self).__init__()

    def call(self, kwargs):
        boxes = kwargs['boxes']
        encoded_boxes = ops.decode(boxes, self.prior_boxes, self.variances)
        kwargs['boxes'] = encoded_boxes
        return kwargs


class NonMaximumSuppressionPerClass(Processor):
    """Applies non maximum suppression per class.
    """
    def __init__(self, nms_thresh=.45, conf_thresh=0.01):
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        super(NonMaximumSuppressionPerClass, self).__init__()

    def call(self, kwargs):
        boxes = kwargs['boxes']
        kwargs['boxes'] = ops.nms_per_class(
            boxes, self.nms_thresh, self.conf_thresh)
        return kwargs


class FilterBoxes(Processor):
    """Filters boxes outputted from function ``detect`` as ``Box2D`` messages
    # Arguments
        class_names: List of strings.
        conf_thresh: Float.
    """
    def __init__(self, class_names, conf_thresh=0.5):
        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.arg_to_class = dict(zip(
            list(range(len(self.class_names))), self.class_names))
        super(FilterBoxes, self).__init__()

    def call(self, kwargs):
        detections = kwargs['boxes']
        kwargs['boxes2D'] = ops.filter_detections(
            detections, self.arg_to_class, self.conf_thresh)
        return kwargs


class PredictBoxes(Processor):
    """TODO: Extend to have pre-processing pipeline.
    """
    def __init__(self, model, mean=BGR_IMAGENET_MEAN):
        self.model = model
        super(PredictBoxes, self).__init__()

    def call(self, kwargs):
        image = ops.resize_image(kwargs['image'], self.model.input_shape[1:3])
        image = image - BGR_IMAGENET_MEAN
        image = np.expand_dims(image, 0)
        kwargs['boxes'] = np.squeeze(self.model.predict(image))
        return kwargs


class ApplyOffsets(Processor):
    def __init__(self, offsets):
        super(ApplyOffsets, self).__init__()
        self.offsets = offsets

    def call(self, kwargs):
        box2D = kwargs['box2D']
        box2D.coordinates = ops.apply_offsets(kwargs['box2D'], self.offsets)
        kwargs['box2D'] = box2D
        return kwargs


class CropImages(Processor):
    def __init__(self, offsets, output_topic='image_crops'):
        self.offsets = offsets
        self.output_topic = output_topic
        super(CropImages, self).__init__()

    def call(self, kwargs):
        x_min, y_min, x_max, y_max = kwargs['box2D'].coordinates
        image_crop = kwargs['image'][y_min:y_max, x_min:x_max]
        kwargs[self.output_topic] = image_crop
        return kwargs
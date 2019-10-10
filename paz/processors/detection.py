import numpy as np

from ..core import Processor, SequentialProcessor, Box2D, ops
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
        topic: String indicating the new key in the data dictionary that
            will contain the list of cropped images.
    """
    def __init__(self, topic='cropped_images'):
        self.topic = topic
        super(CropBoxes2D, self).__init__()

    def call(self, kwargs):
        cropped_images, boxes2D, image = [], kwargs['boxes2D'], kwargs['image']
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_images.append(cropped_image)
        kwargs[self.topic] = cropped_images
        return kwargs


class ToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages.
    # Arguments
        class_names: List of class names ordered with respect to the
            class indices from the dataset `boxes`.
    """
    def __init__(self, class_names, one_hot_encoded=False):
        self.arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        super(ToBoxes2D, self).__init__()

    def call(self, kwargs):
        numpy_boxes2D, boxes2D = kwargs['boxes'], []
        for numpy_box2D in numpy_boxes2D:
            if self.one_hot_encoded:
                class_name = self.arg_to_class[np.argmax(numpy_box2D[4:])]
            else:
                class_name = self.arg_to_class[numpy_box2D[-1]]
            boxes2D.append(Box2D(numpy_box2D[:4], 1.0, class_name))
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
    """Encodes bounding boxes TODO:
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
    """Decodes bounding boxes TODO:
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


class DetectBoxes(Processor):
    """Applies detect function TODO:
    """
    def __init__(self, prior_boxes, nms_thresh=.45, conf_thresh=0.01):
        self.prior_boxes = prior_boxes
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        super(DetectBoxes, self).__init__()

    def call(self, kwargs):
        boxes = kwargs['boxes']
        # print(boxes.shape)
        detections = ops.detect(
            boxes, self.prior_boxes, self.conf_thresh, self.nms_thresh)
        kwargs['boxes'] = detections
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
    def __init__(self, model, mean=BGR_IMAGENET_MEAN):
        self.model = model
        super(PredictBoxes, self).__init__()

    def call(self, kwargs):
        image = ops.resize_image(kwargs['image'], self.model.input_shape[1:3])
        image = image - BGR_IMAGENET_MEAN
        image = np.expand_dims(image, 0)
        kwargs['boxes'] = np.squeeze(self.model.predict(image))
        return kwargs


class DetectBoxes2D(Processor):
    """Detects objects from a single shot model.
    # Arguments
        model: Tensorflow SSD model with prior_boxes.
        class_names: List of strings containing the class names.
        score_thresh: Float between [0, 1]. If the predictions are less
            than this score they are ignored.
        nsm_thresh: Non-maximum supression score.
    """
    def __init__(self, model, class_names, score_thresh=.6, nms_thresh=.45,
                 mean=BGR_IMAGENET_MEAN):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.mean = mean
        self.arg_to_class = dict(zip(
            list(range(self.num_classes)), self.class_names))
        super(DetectBoxes2D, self).__init__()

    def call(self, kwargs):
        args = (kwargs['image'], self.model, self.nms_thresh, self.mean)
        detections = self.detect_from_image(*args)
        args = (detections, self.arg_to_class, self.score_thresh)
        boxes2D = self.filter_detections(*args)
        kwargs['boxes2D'] = boxes2D
        return kwargs

    def detect_from_image(self, image, model, nms_thresh, mean):
        input_shape = model.input_shape[1:3]
        image_array = ops.resize_image(image, input_shape)
        image_array = ops.substract_mean(image_array, mean)
        image_array = np.expand_dims(image_array, 0)
        predictions = model.predict(image_array)
        prior_boxes = model.prior_boxes
        detections = ops.detect(
            predictions, prior_boxes, nms_thresh=nms_thresh)
        return detections

    def filter_detections(self, detections, arg_to_class, conf_thresh=0.5):
        num_classes = detections.shape[0]
        filtered_detections = []
        for class_arg in range(1, num_classes):
            class_detections = detections[class_arg, :]
            confidence_mask = np.squeeze(class_detections[:, -1] > conf_thresh)
            confident_class_detections = class_detections[confidence_mask]

            if len(confident_class_detections) == 0:
                continue

            class_name = arg_to_class[class_arg]
            for confident_class_detection in confident_class_detections:
                coordinates = confident_class_detection[:4]
                score = confident_class_detection[4]
                detection = Box2D(coordinates, score, class_name)
                filtered_detections.append(detection)
        return filtered_detections


class IterateOverCroppedImages(Processor):
    def __init__(self, processes):
        super(IterateOverCroppedImages, self).__init__()
        self.process = SequentialProcessor()
        [self.process.add(process) for process in processes]

    def call(self, kwargs):
        keypoints = []
        images, boxes2D = kwargs['cropped_images'], kwargs['boxes2D']
        for image, box2D in zip(images, boxes2D):
            inferences = self.process(image=image, box2D=box2D)
            keypoints.append(inferences['keypoints'])
        kwargs['keypoints'] = keypoints
        return kwargs

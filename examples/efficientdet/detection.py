import numpy as np
from paz import processors as pr
from paz.backend.image.draw import draw_rectangle
from paz.abstract import SequentialProcessor, Processor
from draw import (compute_text_bounds, draw_opaque_box, make_box_transparent,
                  put_text)
B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV = 57.3 , 57.1, 58.4
RGB_IMAGENET_STDEV = (R_IMAGENET_STDEV, G_IMAGENET_STDEV, B_IMAGENET_STDEV)
from paz.backend.image import resize_image
from paz.processors.image import RGB_IMAGENET_MEAN

# class DrawBoxes2D(pr.DrawBoxes2D):
#     """Draws bounding boxes from Boxes2D messages.

#     # Arguments
#         class_names: List, class names.
#         colors: List, color values.
#         weighted: Bool, whether to weight bounding box color.
#         scale: Float. Scale of text drawn.
#         with_score: Bool, denoting if confidence be shown.

#     # Methods
#         compute_box_color()
#         compute_text()
#         get_text_box_parameters()
#         call()
#     """
#     def __init__(
#             self, class_names=None, colors=None,
#             weighted=False, scale=0.7, with_score=True):
#         super().__init__(
#             class_names, colors, weighted, scale, with_score)

#     def compute_box_color(self, box2D):
#         class_name = box2D.class_name
#         color = self.class_to_color[class_name]
#         if self.weighted:
#             color = [int(channel * box2D.score) for channel in color]
#         return color

#     def compute_text(self, box2D):
#         class_name = box2D.class_name
#         text = '{}'.format(class_name)
#         if self.with_score:
#             text = '{} :{}%'.format(class_name, round(box2D.score * 100))
#         return text

#     def get_text_box_parameters(self):
#         thickness = 1
#         offset_x = 2
#         offset_y = 17
#         color = (0, 0, 0)
#         text_parameters = [thickness, offset_x, offset_y, color]
#         box_start_offset = 2
#         box_end_offset = 5
#         box_color = (255, 174, 66)
#         text_box_parameters = [box_start_offset, box_end_offset, box_color]
#         return [text_box_parameters, text_parameters]

#     def call(self, image, boxes2D):
#         raw_image = image.copy()
#         for box2D in boxes2D:
#             x_min, y_min, x_max, y_max = box2D.coordinates
#             color = self.compute_box_color(box2D)
#             draw_opaque_box(image, (x_min, y_min), (x_max, y_max), color)
#         image = make_box_transparent(raw_image, image)
#         text_box_parameters, text_parameters = self.get_text_box_parameters()
#         offset_start, offset_end, text_box_color = text_box_parameters
#         text_thickness, offset_x, offset_y, text_color = text_parameters
#         for box2D in boxes2D:
#             x_min, y_min, x_max, y_max = box2D.coordinates
#             color = self.compute_box_color(box2D)
#             draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
#             text = self.compute_text(box2D)
#             text_size = compute_text_bounds(text, self.scale, text_thickness)
#             (text_W, text_H), _ = text_size
#             args = (image, (x_min + offset_start, y_min + offset_start),
#                     (x_min + text_W + offset_end, y_min + text_H + offset_end),
#                     text_box_color)
#             draw_opaque_box(*args)
#             args = (image, text, (x_min + offset_x, y_min + offset_y),
#                     self.scale, text_color, text_thickness)
#             put_text(*args)
#         return image


class DetectSingleShotEfficientDet(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN, variances=[1, 1, 1, 1],
                 draw=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw

        super(DetectSingleShotEfficientDet, self).__init__()
        # preprocessing = SequentialProcessor(
        #     [pr.ResizeImage(self.model.input_shape[1:3]),
        #      pr.ConvertColorSpace(pr.RGB2BGR),
        #      pr.SubtractMeanImage(mean),
        #      pr.CastImage(float),
        #      pr.ExpandDims(axis=0)])

        preprocessing = SequentialProcessor([
            pr.CastImage(float),
            pr.SubtractMeanImage(mean=RGB_IMAGENET_MEAN),
            DivideStandardDeviationImage(standard_deviation=RGB_IMAGENET_STDEV),
            ScaledResize(image_size=self.model.input_shape[1]),
        ])
        self.preprocessing = preprocessing

        # postprocessing = SequentialProcessor(
        #     [pr.Squeeze(axis=None),
        #      pr.DecodeBoxes(self.model.prior_boxes, self.variances),
        #      pr.NonMaximumSuppressionPerClass(self.nms_thresh),
        #      pr.FilterBoxes(self.class_names, self.score_thresh)])
        

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        preprocessed_image, image_scales = self.preprocessing(image)
        postprocessing = SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(self.model.prior_boxes*self.model.input_shape[1],
                           variances=self.variances),
            ScaleBox(image_scales),
            pr.NonMaximumSuppressionPerClass(self.nms_thresh),
            pr.FilterBoxes(get_class_name_efficientdet('COCO'),
                           self.score_thresh)])
        outputs = self.model(preprocessed_image)
        outputs = process_outputs(outputs)
        boxes2D = postprocessing(outputs)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


def get_class_name_efficientdet(dataset_name):
    if dataset_name == 'COCO':
        return ['person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '0', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', '0', 'backpack', 'umbrella', '0',
                '0', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', '0', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', '0', 'dining table', '0', '0',
                'toilet', '0', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '0', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    elif dataset_name == 'VOC':
        return ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def process_outputs(outputs):
    """Merges all feature levels into single tensor and combines box offsets
    and class scores.

    # Arguments
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        num_levels: Int, number of levels considered at efficientnet features.
        num_classes: Int, number of classes in the dataset.

    # Returns
        outputs: Numpy array, Processed outputs by merging the features at
        all levels. Each row corresponds to box coordinate offsets and
        sigmoid of the class logits.
    """
    outputs = outputs[0]
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = np.hsplit(boxes, 4)
    boxes = np.concatenate([s2, s1, s4, s3], axis=1)
    boxes = boxes[np.newaxis]
    classes = classes[np.newaxis]
    outputs = np.concatenate([boxes, classes], axis=2)
    return outputs


def scale_box(predictions, image_scales=None):
    """
    # Arguments
        image: Numpy array.
        boxes: Numpy array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        Numpy array of shape `[num_boxes, N]`.
    """

    if image_scales is not None:
        boxes = predictions[:, :4]
        scales = image_scales[np.newaxis][np.newaxis]
        boxes = boxes * scales
        predictions = np.concatenate([boxes, predictions[:, 4:]], 1)
    return predictions


class ScaleBox(Processor):
    """Scale box coordinates of the prediction.
    """
    def __init__(self, scales):
        super(ScaleBox, self).__init__()
        self.scales = scales

    def call(self, boxes):
        boxes = scale_box(boxes, self.scales)
        return boxes


def efficientdet_preprocess(image, image_size):
    """Preprocess image for EfficientDet model.

    # Arguments
        image: Tensor, raw input image to be preprocessed
        of shape [bs, h, w, c]
        image_size: Tensor, size to resize the raw image
        of shape [bs, new_h, new_w, c]

    # Returns
        image: Numpy array, resized and preprocessed image
        image_scale: Numpy array, scale to reconstruct each of
        the raw images to original size from the resized
        image.
    """

    preprocessing = SequentialProcessor([
        pr.CastImage(float),
        pr.SubtractMeanImage(mean=RGB_IMAGENET_MEAN),
        DivideStandardDeviationImage(standard_deviation=RGB_IMAGENET_STDEV),
        ScaledResize(image_size=image_size),
        ])
    image, image_scale = preprocessing(image)
    return image, image_scale

class DivideStandardDeviationImage(Processor):
    """Divide channel-wise standard deviation to image.

    # Arguments
        mean: List of length 3, containing the channel-wise mean.
    """
    def __init__(self, standard_deviation):
        self.standard_deviation = standard_deviation
        super(DivideStandardDeviationImage, self).__init__()

    def call(self, image):
        return image / self.standard_deviation

class ScaledResize(Processor):
    """Resizes image by returning the scales to original image.

    # Arguments
        image_size: Int, desired size of the model input.

    # Returns
        output_images: Numpy array, image resized to match
        image size.
        image_scales: Numpy array, scale to reconstruct the
        raw image from the output_images.
    """
    def __init__(self, image_size):
        self.image_size = image_size
        super(ScaledResize, self).__init__()

    def call(self, image):
        """
        # Arguments
            image: Numpy array, raw input image.
        """
        crop_offset_y = np.array(0)
        crop_offset_x = np.array(0)
        height = np.array(image.shape[0]).astype('float32')
        width = np.array(image.shape[1]).astype('float32')
        image_scale_y = np.array(self.image_size).astype('float32') / height
        image_scale_x = np.array(self.image_size).astype('float32') / width
        image_scale = np.minimum(image_scale_x, image_scale_y)
        scaled_height = (height * image_scale).astype('int32')
        scaled_width = (width * image_scale).astype('int32')
        scaled_image = resize_image(image, (scaled_width, scaled_height))
        scaled_image = scaled_image[
                       crop_offset_y: crop_offset_y + self.image_size,
                       crop_offset_x: crop_offset_x + self.image_size,
                       :]
        output_images = np.zeros((self.image_size,
                                  self.image_size,
                                  image.shape[2]))
        output_images[:scaled_image.shape[0],
        :scaled_image.shape[1],
        :scaled_image.shape[2]] = scaled_image
        image_scale = 1 / image_scale
        output_images = output_images[np.newaxis]
        return output_images, image_scale


def efficientdet_postprocess(model, outputs, image_scales, raw_images=None):
    """EfficientDet output postprocessing function.

    # Arguments
        model: EfficientDet model
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        image_scale: Numpy array, scale to reconstruct each of the raw images
        to original size from the resized image.
        raw_images: Numpy array, RGB image to draw the detections on the image.

    # Returns
        image: Numpy array, RGB input image with detections overlaid.
        outputs: List of Box2D, containing the detections with bounding box
        and class details.
    """
    outputs = process_outputs(outputs)
    postprocessing = SequentialProcessor(
        [pr.Squeeze(axis=None),
         pr.DecodeBoxes(model.prior_boxes*512, variances=[1, 1, 1, 1]),
         ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
         pr.FilterBoxes(get_class_name_efficientdet('COCO'), 0.8)])
    outputs = postprocessing(outputs)
    draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('COCO'))
    image = draw_boxes2D(raw_images.astype('uint8'), outputs)
    return image, outputs
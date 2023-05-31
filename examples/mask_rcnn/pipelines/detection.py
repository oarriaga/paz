import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Lambda

from paz.abstract import Processor
from mask_rcnn.backend.boxes import norm_boxes, denorm_boxes
from paz.backend.image.opencv_image import resize_image
from paz.backend.image.image import cast_image

from mask_rcnn.pipelines.data_generator import compute_backbone_shapes
from mask_rcnn.backend.image import subtract_mean_image, generate_original_masks
from mask_rcnn.backend.boxes import generate_pyramid_anchors


class NormalizeImages(Processor):
    def __init__(self):
        super(NormalizeImages, self).__init__()

    def call(self, images, windows, pixel_mean=np.array([123.7, 116.8, 103.9])):
        normalized_images = []
        for image in images:
            molded_image = subtract_mean_image(image, pixel_mean)
            normalized_images.append(molded_image)
        return normalized_images, windows


class ResizeImages(Processor):
    def __init__(self):
        super(ResizeImages, self).__init__()

    def call(self, images):
        resized_images, windows = [], []
        for image in images:
            H, W = image.shape[:2]
            resized_image = resize_image(image, (H, W))
            resized_image = cast_image(resized_image, 'uint8')
            window = (0, 0, H, W)

            resized_images.append(resized_image)
            windows.append(window)
        return resized_images, windows


class Detect(Processor):
    def __init__(self, model, anchor_scales, batch_size, preprocess=None, postprocess=None):
        self.base_model = model
        self.model = model.keras_model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.anchor_scales = anchor_scales
        self.batch_size = batch_size

    def call(self, images):
        normalized_images, windows = self.preprocess(images)
        image_shape = normalized_images[0].shape

        anchors = self.get_anchors(image_shape, np.array(images))
        anchors = np.broadcast_to(anchors,
                                  (self.batch_size,) + anchors.shape)
        detections, predicted_classes, mrcnn_bounding_box, predicted_masks, rpn_rois, rpn_class, rpn_bounding_box = \
            self.model.predict([normalized_images, anchors])

        results = self.postprocess(images, normalized_images, windows,
                                   detections, predicted_masks)

        return results

    def get_anchors(self, image_shape, images):
        backbone_shapes = compute_backbone_shapes(backbone="resnet101", image_shape=image_shape)
        if not hasattr(self, '_anchor_cache'):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            anchors = generate_pyramid_anchors(
                self.anchor_scales,
                [0.5, 1, 2],
                backbone_shapes,
                [4, 8, 16, 32, 64], 1)

            self.anchors = anchors.copy()
            self.anchors[:, 0], self.anchors[:, 1], self.anchors[:, 2], self.anchors[:, 3] = \
                anchors[:, 1], anchors[:, 0], anchors[:, 3], anchors[:, 2]

            self._anchor_cache[tuple(image_shape)] = norm_boxes(
                self.anchors, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]


class PostprocessInputs(Processor):
    def __init__(self):
        super(PostprocessInputs, self).__init__()

    def call(self, images, normalized_images, windows, detections, predicted_masks):
        results = []
        for index, image in enumerate(images):
            boxes, class_ids, scores, masks = self.postprocess(
                detections[index], predicted_masks[index],
                image.shape, normalized_images[index].shape, windows[index])
            results.append({
                'rois': boxes,
                'class_ids': class_ids,
                'scores': scores,
                'masks': masks,
            })
        return results

    def postprocess(self, detections, predicted_masks, original_image_shape,
                    image_shape, window):
        zero_index = np.where(detections[:, 4] == 0)[0]
        N = zero_index[0] if zero_index.shape[0] > 0 else detections.shape[0]

        boxes, class_ids, scores, masks = self.unpack_detections(
            N, detections, predicted_masks)
        boxes = self.normalize_boxes(boxes, window, image_shape,
                                     original_image_shape)
        boxes, class_ids, scores, masks, N = self.filter_detections(
            N, boxes, class_ids, scores, masks)
        full_masks = self.unmold_masks(N, boxes, masks, original_image_shape)
        return boxes, class_ids, scores, full_masks

    def unpack_detections(self, N, detections, predicted_masks):
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = predicted_masks[np.arange(N), :, :, class_ids]
        return boxes, class_ids, scores, masks

    def normalize_boxes(self, boxes, window, image_shape, original_image_shape):
        window = norm_boxes(window, image_shape[:2])
        Wy_min, Wx_min, Wy_max, Wx_max = window
        shift = np.array([Wy_min, Wx_min, Wy_min, Wx_min])
        window_H = Wy_max - Wy_min
        window_W = Wx_max - Wx_min
        scale = np.array([window_H, window_W, window_H, window_W])
        boxes = np.divide(boxes - shift, scale)
        boxes = denorm_boxes(boxes, original_image_shape[:2])
        return boxes

    def filter_detections(self, N, boxes, class_ids, scores, masks):
        exclude_index = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_index.shape[0] > 0:
            boxes = np.delete(boxes, exclude_index, axis=0)
            class_ids = np.delete(class_ids, exclude_index, axis=0)
            scores = np.delete(scores, exclude_index, axis=0)
            masks = np.delete(masks, exclude_index, axis=0)
            N = class_ids.shape[0]
        return boxes, class_ids, scores, masks, N

    def unmold_masks(self, N, boxes, masks, original_image_shape):
        full_masks = []
        for index in range(N):
            boxes_xy = [boxes[index, 1], boxes[index, 0], boxes[index, 3], boxes[index, 2]]
            full_mask = generate_original_masks(masks[index], boxes_xy,
                                                original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else \
            np.empty(original_image_shape[:2] + (0,))
        return full_masks

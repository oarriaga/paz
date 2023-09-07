import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Lambda

from paz.abstract import Processor
from paz.backend.image.opencv_image import resize_image
from paz.backend.image.image import cast_image

from mask_rcnn.pipelines.data_generator import ComputeBackboneShapes
from mask_rcnn.backend.image import subtract_mean_image, cast_image
from mask_rcnn.backend.image import resize_to_original_size
from mask_rcnn.backend.boxes import generate_pyramid_anchors
from mask_rcnn.backend.boxes import normalized_boxes, denormalized_boxes


class NormalizeImages(Processor):
    def __init__(self):
        super(NormalizeImages, self).__init__()

    def call(self, images, windows,
             pixel_mean=np.array([123.7, 116.8, 103.9])):
        normalized_images = []
        for image in images:
            # image = cast_image(image, "float32")
            molded_image = subtract_mean_image(image, pixel_mean)
            normalized_images.append(molded_image)
        return normalized_images, windows


class ResizeImages(Processor):
    def __init__(self, min_dim=800, scale=0, max_dim=1024):
        super(ResizeImages, self).__init__()
        self.min_dim = min_dim
        self.scale = scale
        self.max_dim = max_dim

    def call(self, images):
        resized_images, windows = [], []
        for image in images:
            resized_image, window = self.resize(image, min_dim=self.min_dim,
                                                min_scale=self.scale,
                                                max_dim=self.max_dim)
            resized_images.append(resized_image)
            windows.append(window)
        return resized_images, windows

    def resize(self, image, min_dim=None, max_dim=None, min_scale=None,
               scale=1):
        image_dtype = image.dtype
        H, W = image.shape[:2]

        if min_dim:
            scale = max(1, min_dim / min(H, W))
        if min_scale and scale < min_scale:
            scale = min_scale
        if max_dim:
            image_max = max(H, W)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max
        if scale != 1:
            image = resize_image(image, (round(W * scale), round(H * scale)))

        padded_image, window = self.pad_image(image, max_dim)
        return padded_image.astype(image_dtype), window

    def pad_image(self, image, max_dim):
        H, W = image.shape[:2]
        top_pad = (max_dim - H) // 2
        bottom_pad = max_dim - H - top_pad
        left_pad = (max_dim - W) // 2
        right_pad = max_dim - W - left_pad

        padding = [(bottom_pad, top_pad), (right_pad, left_pad), (0, 0)]
        padded_image = np.pad(image, padding, mode='constant',
                              constant_values=0)
        window = (bottom_pad, right_pad, H + bottom_pad, W + right_pad)
        return padded_image, window


class Detect(Processor):
    def __init__(self, model, anchor_scales, batch_size, preprocess=None,
                 postprocess=None):
        self.base_model = model
        self.model = model.keras_model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.anchor_scales = anchor_scales
        self.batch_size = batch_size

    def call(self, images):
        normalized_images, windows = self.preprocess(images)
        image_shape = normalized_images[0].shape

        anchors = self.get_anchors(image_shape[:2])
        anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
        detections, _, _, predicted_masks, _, _, _ = self.model.predict(
            [normalized_images, anchors])
        results = self.postprocess(images, normalized_images, windows,
                                   detections, predicted_masks)
        return results

    def get_anchors(self, image_shape):
        backbone_shapes = ComputeBackboneShapes()(backbone="resnet101",
                                                  image_shape=image_shape)
        if not hasattr(self, '_anchor_cache'):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            self.anchors = generate_pyramid_anchors(self.anchor_scales,
                                                    [0.5, 1, 2],
                                                    backbone_shapes,
                                                    [4, 8, 16, 32, 64], 1)
            self._anchor_cache[tuple(image_shape)] = normalized_boxes(
                self.anchors, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]


class PostprocessInputs(Processor):
    def __init__(self):
        super(PostprocessInputs, self).__init__()

    def call(self, images, normalized_images, windows, detections,
             predicted_masks):
        results = []
        for index, image in enumerate(images):
            boxes, class_ids, scores, masks = self.postprocess(
                detections[index], predicted_masks[index],
                image.shape, normalized_images[index].shape,
                windows[index])
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

    def normalize_boxes(self, boxes, window, image_shape,
                        original_image_shape):
        window = normalized_boxes(window, image_shape[:2])
        print("window", window)
        Wy_min, Wx_min, Wy_max, Wx_max = window
        shift = np.array([Wy_min, Wx_min, Wy_min, Wx_min])
        window_H = Wy_max - Wy_min
        window_W = Wx_max - Wx_min
        scale = np.array([window_H, window_W, window_H, window_W])
        boxes = np.divide(boxes - shift, scale)
        boxes = denormalized_boxes(boxes, original_image_shape[:2])
        print("boxes", boxes)
        return boxes

    def filter_detections(self, N, boxes, class_ids, scores, masks):
        exclude_index = np.where(
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1]) <= 100)[0]
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
            full_mask = resize_to_original_size(masks[index], boxes[index],
                                                original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(
            original_image_shape[:2] + (0,))
        return full_masks

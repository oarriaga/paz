import numpy as np
from paz.abstract import Processor
from mask_rcnn.utils import resize_image, normalize_image
from mask_rcnn.utils import compute_backbone_shapes, norm_boxes
from mask_rcnn.utils import generate_pyramid_anchors, denorm_boxes
from mask_rcnn.utils import unmold_mask


class NormalizeImages(Processor):
    def __init__(self, config):
        self.config = config
        super(NormalizeImages, self).__init__()

    def call(self, images, windows):
        normalized_images = []
        for image in images:
            molded_image = normalize_image(image, self.config)
            normalized_images.append(molded_image)
        return normalized_images, windows


class ResizeImages(Processor):
    def __init__(self, config):
        self.IMAGE_MIN_DIM = config.IMAGE_MIN_DIM
        self.IMAGE_MIN_SCALE = config.IMAGE_MIN_SCALE
        self.IMAGE_MAX_DIM = config.IMAGE_MAX_DIM
        self.IMAGE_RESIZE_MODE = config.IMAGE_RESIZE_MODE

    def call(self, images):
        resized_images, windows = [], []
        for image in images:
            resized_image, window, _, _, _ = resize_image(
                image,
                min_dim=self.IMAGE_MIN_DIM,
                min_scale=self.IMAGE_MIN_SCALE,
                max_dim=self.IMAGE_MAX_DIM,
                mode=self.IMAGE_RESIZE_MODE)
            resized_images.append(resized_image)
            windows.append(window)
        return resized_images, windows


class Detect(Processor):
    def __init__(self, model, config, preprocess=None, postprocess=None):
        self.base_model = model
        self.model = model.keras_model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.config = config

    def call(self, images):
        normalized_images, windows = self.preprocess(images)
        image_shape = normalized_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors,
                                  (self.config.BATCH_SIZE,) + anchors.shape)
        detections, _, _, predicted_masks, _, _, _ =\
            self.model.predict([normalized_images, anchors])
        results = self.postprocess(images, normalized_images, windows,
                                   detections, predicted_masks)
        return results

    def get_anchors(self, image_shape):
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, '_anchor_cache'):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            anchors = generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = anchors
            self._anchor_cache[tuple(image_shape)] = norm_boxes(
                anchors, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]


class PostprocessInputs(Processor):
    def __init__(self):
        super(PostprocessInputs, self).__init__()

    def call(self, images, normalized_images, windows,
             detections, predicted_masks):
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

    def normalize_boxes(self, boxes, window, image_shape,
                        original_image_shape):
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
            full_mask = unmold_mask(masks[index], boxes[index],
                                    original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))
        return full_masks

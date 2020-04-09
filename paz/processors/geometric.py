from ..core import Processor
from ..core import backend as pz

import tensorflow as tf


class RandomFlipBoxesLeftRight(Processor):
    """Flips image and implemented labels horizontally.
    Current implemented labels include ``boxes``.
    """
    def __init__(self):
        super(RandomFlipBoxesLeftRight, self).__init__()

    def call(self, image, boxes):
        if tf.random.uniform([1], 0, 2, tf.int32) == 1:
            boxes = pz.boxes.flip_left_right(boxes, image.shape[1])
            image = pz.image.flip_left_right(image)
        return image, boxes


class ToAbsoluteCoordinates(Processor):
    """Convert normalized box coordinates to image box coordinates.
    """
    def __init__(self):
        super(ToAbsoluteCoordinates, self).__init__()

    def call(self, image, boxes):
        boxes = pz.boxes.to_absolute_coordinates(image, boxes)
        return image, boxes


class ToPercentCoordinates(Processor):
    """Convert image box coordinates to normalized box coordinates.
    """
    def __init__(self):
        super(ToPercentCoordinates, self).__init__()

    def call(self, image, boxes):
        boxes = pz.boxes.to_percent_coordinates(image, boxes)
        return image, boxes


class RandomSampleCrop(Processor):
    """Crops and image while adjusting the bounding boxes.
    Boxes should be in point form.
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        super(RandomSampleCrop, self).__init__()

    def call(self, image, boxes=None):
        boxes = boxes[:, :4]
        labels = boxes[:, -1:]
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                boxes = np.hstack([boxes, labels])
                return image, boxes

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = ops.compute_iou(rect, boxes)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                return current_image, np.hstack(
                    [current_boxes, current_labels])


class Expand(Processor):
    """Expand image size up to 2x, 3x, 4x and fill values with mean color.
    This transformation is applied with a probability of 50%.
    # Arguments
        max_ratio: Float.
        mean: None/List: If `None` expanded image is filled with
            the image mean.
    """
    def __init__(self, max_ratio=2, probability=0.5, mean=None):
        super(Expand, self).__init__(probability)
        self.max_ratio = max_ratio
        self.mean = mean

    def call(self, image, boxes=None, keypoints=None):
        height, width, num_channels = image.shape
        ratio = np.random.uniform(1, self.max_ratio)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)
        expanded_image = np.zeros((int(height * ratio),
                                   int(width * ratio), num_channels),
                                  dtype=image.dtype)

        if self.mean is None:
            expanded_image[:, :, :] = np.mean(image, axis=(0, 1))
        else:
            expanded_image[:, :, :] = self.mean

        expanded_image[int(top):int(top + height),
                       int(left):int(left + width)] = image

        if boxes is not None:
            boxes[:, 0:2] = boxes[:, 0:2] + (int(left), int(top))
            boxes[:, 2:4] = boxes[:, 2:4] + (int(left), int(top))
            return expanded_image, boxes

        if keypoints is not None:
            keypoints[:, :2] = keypoints[:, :2] + (int(left), int(top))
            return expanded_image, keypoints
        return expanded_image


class ApplyTranslation(Processor):
    """Applies a translation of image and labels
    # Arguments
        translation: A list of length two indicating the x,y translation values
        fill_color: List of three integers indicating the
            color values e.g. [0,0,0]
    # TODO:
        Implement function for ``boxes`` labels
    """
    def __init__(self, translation, fill_color=None):
        super(ApplyTranslation, self).__init__()
        self._matrix = np.zeros((2, 3), dtype=np.float32)
        self._matrix[0, 0], self._matrix[1, 1] = 1.0, 1.0
        self.fill_color = fill_color
        self.translation = translation

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        if translation is None:
            self._translation = None
        elif len(translation) == 2:
            self._translation = translation
            self._matrix[0, 2], self._matrix[1, 2] = translation
        else:
            raise ValueError('Translation should be `None` or have length two')

    def call(self, image, keypoints=None):
        height, width = image.shape[:2]
        if self.fill_color is None:
            fill_color = np.mean(image, axis=(0, 1))
        image = ops.warp_affine(image, self._matrix, fill_color)
        if keypoints is not None:
            keypoints[:, 0] = keypoints[:, 0] + self.translation[0]
            keypoints[:, 1] = keypoints[:, 1] + self.translation[1]
            return image, keypoints
        return image


class ApplyRandomTranslation(Processor):
    """Applies a random translation to image and labels
    #Arguments
        delta_scale: List with two elements having the normalized deltas.
            e.g. [.25, .25]
        fill_color: List of three integers indicating the
            color values e.g. [0,0,0]
        probability: Float between [0, 1]
    """
    def __init__(
            self, delta_scale=[0.25, 0.25], fill_color=None, probability=0.5):
        super(ApplyRandomTranslation, self).__init__(probability)
        self.delta_scale = delta_scale
        self.apply_translation = ApplyTranslation(None, fill_color)

    @property
    def delta_scale(self):
        return self._delta_scale

    @delta_scale.setter
    def delta_scale(self, delta_scale):
        x_delta_scale, y_delta_scale = delta_scale
        if (x_delta_scale < 0) or (y_delta_scale < 0):
            raise ValueError('Delta scale values should be a positive scalar')
        self._delta_scale = delta_scale

    def call(self, image):
        height, width = image.shape[:2]
        x_delta_scale, y_delta_scale = self.delta_scale
        x = image.shape[1] * np.random.uniform(-x_delta_scale, x_delta_scale)
        y = image.shape[0] * np.random.uniform(-y_delta_scale, y_delta_scale)
        self.apply_translation.translation = [x, y]
        return self.apply_translation(image)

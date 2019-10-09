from ..core import Processor
from ..core import ops
import numpy as np


class HorizontalFlip(Processor):
    """Flips image and implemented labels horizontally.
    Current implemented labels include ``boxes``.
    """
    def __init__(self, probability=0.5):
        super(HorizontalFlip, self).__init__(probability)

    def call(self, kwargs):
        if 'image' in kwargs:
            image = kwargs['image']
            width = image.shape[1]
            image = image[:, ::-1]
            kwargs['image'] = image

        if 'boxes' in kwargs:
            box_data = kwargs['boxes']
            kwargs['boxes'][:, [0, 2]] = width - box_data[:, [2, 0]]

        if 'keypoints' in kwargs:
            raise NotImplementedError

        return kwargs


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

    def call(self, kwargs):
        image = kwargs['image']
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
        kwargs['image'] = expanded_image

        if 'boxes' in kwargs:
            boxes = kwargs['boxes']
            boxes[:, 0:2] = boxes[:, 0:2] + (int(left), int(top))
            boxes[:, 2:4] = boxes[:, 2:4] + (int(left), int(top))
            kwargs['boxes'] = boxes

        if 'keypoints' in kwargs:
            keypoints = kwargs['keypoints']
            keypoints[:, :2] += (int(left), int(top))
            # there was a BUG when ran with complete pipeline
            # if [:, :2] was not there the KP were not showing
            kwargs['keypoints'][:, :2] = keypoints[:, :2]
        return kwargs


class ToAbsoluteCoordinates(Processor):
    """Convert normalized box coordinates to image box coordinates.
    """
    def __init__(self):
        super(ToAbsoluteCoordinates, self).__init__()

    def call(self, kwargs):
        height, width, channels = kwargs['image'].shape
        boxes = kwargs['boxes']
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        kwargs['boxes'] = boxes
        return kwargs


class ToPercentCoordinates(Processor):
    """Convert image box coordinates to normalized box coordinates.
    """

    def __init__(self):
        super(ToPercentCoordinates, self).__init__()

    def call(self, kwargs):
        height, width, channels = kwargs['image'].shape
        boxes = kwargs['boxes']
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        kwargs['boxes'] = boxes
        return kwargs


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

    def call(self, kwargs):
        image = kwargs['image']
        boxes = kwargs['boxes'][:, :4]
        labels = kwargs['boxes'][:, -1:]
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                kwargs['image'] = image
                kwargs['boxes'] = np.hstack([boxes, labels])
                return kwargs

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
                kwargs['image'] = current_image
                kwargs['boxes'] = np.hstack([current_boxes, current_labels])
                return kwargs

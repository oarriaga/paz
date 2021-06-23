import numpy as np
from paz.abstract import Processor
from paz.backend.image import blend_alpha_channel, make_random_plain_image

from backend import load_image, match, random_shape_crop


class LoadImage(Processor):
    """Loads image.

    # Arguments
        num_channels: Integer, valid integers are: 1, 3 and 4.
    """
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        super(LoadImage, self).__init__()

    def call(self, image):
        return load_image(image, self.num_channels)


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


class BlendRandomCroppedBackground(Processor):
    """Blends image with a randomly cropped background.

    # Arguments
        background_paths: List of strings. Each element of the list is a
            full-path to an image used for cropping a background.
    """
    def __init__(self, background_paths):
        super(BlendRandomCroppedBackground, self).__init__()
        if not isinstance(background_paths, list):
            raise ValueError('``background_paths`` must be list')
        if len(background_paths) == 0:
            raise ValueError('No paths given in ``background_paths``')
        self.background_paths = background_paths

    def call(self, image):
        random_arg = np.random.randint(0, len(self.background_paths))
        background_path = self.background_paths[random_arg]
        background = load_image(background_path)
        background = random_shape_crop(background, image.shape[:2])
        if background is None:
            H, W, num_channels = image.shape
            # background contains always a channel less
            num_channels = num_channels - 1
            background = make_random_plain_image((H, W, num_channels))
        return blend_alpha_channel(image, background)

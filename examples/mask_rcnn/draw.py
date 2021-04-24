import numpy as np
from paz.processors import Processor
from paz.backend.image import random_colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply given mask to the image
    """
    for channel in range(3):
        image[:, :, channel] = np.where(mask == 1,
                                        image[:, :, channel] *
                                        (1 - alpha) +
                                        alpha * color[channel] * 255,
                                        image[:, :, channel])
    return image


class DrawBinaryMasks(Processor):
    """ Draws Binary mask on an image
    """
    def __init__(self):
        super(DrawBinaryMasks, self).__init__()

    def call(self, image, masks):
        colors = random_colors(masks.shape[-1])
        masked = image.copy()
        for index in range(masks.shape[-1]):
            mask = masks[:, :, index]
            masked = apply_mask(masked, mask, colors[index])
        return masked

from paz import processors as pr
from paz.backend.image.draw import lincolor
# from paz.backend.image import show_image
from backend import resize_image_with_nearest_neighbors
import numpy as np


class PreprocessImage(pr.SequentialProcessor):
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.add(pr.SubtractMeanImage(mean))


CITY_ESCAPES_ID_TO_MASK = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1,
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 3, 18: 3, 19: 3, 20: 3,
    21: 4, 22: 4, 23: 5, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7, 29: 7, 30: 7,
    31: 7, 32: 7, 33: 7, -1: 7}


class FromIdToMask(pr.Processor):
    def __init__(self, id_to_mask=CITY_ESCAPES_ID_TO_MASK):
        super(FromIdToMask, self).__init__()
        self.id_to_mask = id_to_mask
        self.num_classes = len(set(list(self.id_to_mask.values())))

    def call(self, image):
        H, W = image.shape[:2]
        masks = np.zeros((H, W, self.num_classes))
        unique_ids = np.unique(image)
        for unique_id in unique_ids:
            mask_arg = self.id_to_mask[unique_id]
            id_mask = image[:, :, 0] == unique_id
            mask = masks[:, :, mask_arg]
            mask = mask + id_mask.astype('int')
            masks[:, :, mask_arg] = mask
        return masks


class ResizeImageWithNearestNeighbors(pr.Processor):
    def __init__(self, shape):
        self.shape = shape
        super(ResizeImageWithNearestNeighbors, self).__init__()

    def call(self, image):
        return resize_image_with_nearest_neighbors(image, self.shape)


class MasksToColors(pr.Processor):
    def __init__(self, num_classes, colors=None):
        super(MasksToColors, self).__init__()
        self.num_classes = num_classes
        self.colors = colors
        if self.colors is None:
            self.colors = lincolor(self.num_classes, normalized=True)

    def call(self, masks):
        H, W, num_masks = masks.shape
        assert num_masks == self.num_classes
        image = np.zeros((H, W, 3))
        for mask_arg in range(self.num_classes):
            mask = masks[..., mask_arg]
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
            color = self.colors[mask_arg]
            color_mask = color * mask
            # image = (image + color_mask) / 2.0
            image = image + color_mask
            # show_image((255 * image).astype('uint8'))
        return image


class Round(pr.Processor):
    def __init__(self, decimals=0):
        super(Round, self).__init__()
        self.decimals = decimals

    def call(self, image):
        return np.round(image, self.decimals)

from paz import processors as pr
from paz.backend.image.draw import lincolor
import numpy as np


class PreprocessImage(pr.SequentialProcessor):
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.add(pr.SubtractMeanImage(mean))


class PreprocessSegmentation(pr.SequentialProcessor):
    def __init__(self, image_shape, num_classes, input_name='input_1'):
        super(PreprocessSegmentation, self).__init__()
        H, W = image_shape
        preprocess_image = PreprocessImage()
        self.add(pr.UnpackDictionary(['image', 'masks']))
        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.SequenceWrapper({0: {input_name: [H, W, 3]}},
                                    {1: {'masks': [H, W, num_classes]}}))


class MasksToColors(pr.Processor):
    def __init__(self, num_classes, colors=None):
        super(MasksToColors, self).__init__()
        self.num_classes = num_classes
        self.colors = colors
        if self.colors is None:
            self.colors = lincolor(self.num_classes, normalized=True)

    def call(self, masks):
        H, W, num_masks = masks.shape
        image = np.zeros((H, W, 3))
        for mask_arg in range(self.num_classes):
            mask = masks[..., mask_arg]
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
            color = self.colors[mask_arg]
            color_mask = color * mask
            # image = (image + color_mask) / 2.0
            image = image + color_mask
        return image


class Round(pr.Processor):
    def __init__(self, decimals=0):
        super(Round, self).__init__()
        self.decimals = decimals

    def call(self, image):
        return np.round(image, self.decimals)

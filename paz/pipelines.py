from .core import SequentialProcessor
from . import processors as pr


class DetectionAugmentation(SequentialProcessor):
    def __init__(self, split='train', size=300, mean=pr.BGR_IMAGENET_MEAN):
        super(DetectionAugmentation, self).__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split mode')

        self.mean, self.size, self.split = mean, size, split
        if self.split == 'train':
            self.add(pr.CastImageToFloat())
            self.add(pr.RandomContrast())
            self.add(pr.RandomBrightness())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor(transform='HSV'))
            self.add(pr.CastImageToFloat())
            self.add(pr.ToAbsoluteCoords())
            self.add(pr.RandomSaturation())
            self.add(pr.RandomHue())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('HSV', transform='BGR'))
            self.add(pr.Expand(self.mean))
            self.add(pr.RandomSampleCrop())
            self.add(pr.HorizontalFlip())
            self.add(pr.ToPercentCoords())
            self.add(pr.RandomLightingNoise())
            self.add(pr.Resize(self.size))
            self.add(pr.CastImageToFloat())
            self.add(pr.SubtractMeans(self.mean))

        elif ((self.split == 'val') or (self.split == 'test')):
            self.add(pr.CastImageToFloat())
            self.add(pr.Resize(self.size))
            self.add(pr.SubtractMeans(self.mean))


class SingleShotInference(SequentialProcessor):
    def __init__(self, model, class_names, score_thresh, nms_thresh):
        super(SingleShotInference, self).__init__()
        args = (model, class_names, score_thresh, nms_thresh)
        self.add(pr.DetectBoxes2D(*args))
        self.add(pr.DenormalizeBoxes2D())
        self.add(pr.DrawBoxes2D(class_names))

from .core import SequentialProcessor
from . import processors as pr


class DetectionAugmentation(SequentialProcessor):
    def __init__(self, prior_boxes, num_classes, split='train', size=300,
                 iou=.5, variances=[.1, .2], mean=pr.BGR_IMAGENET_MEAN):

        super(DetectionAugmentation, self).__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split mode')
        self.mean, self.size, self.split = mean, size, split
        self.prior_boxes, self.num_classes = prior_boxes, num_classes

        if self.split == 'train':
            self.add(pr.LoadImage())
            self.add(pr.CastImageToFloat())
            self.add(pr.RandomContrast())
            self.add(pr.RandomBrightness())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('BGR', to='HSV'))
            self.add(pr.CastImageToFloat())
            self.add(pr.ToAbsoluteCoordinates())
            self.add(pr.RandomSaturation())
            self.add(pr.RandomHue())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('HSV', to='BGR'))
            self.add(pr.RandomLightingNoise())
            self.add(pr.Expand(mean=self.mean))
            self.add(pr.RandomSampleCrop())
            self.add(pr.HorizontalFlip())
            self.add(pr.ToPercentCoordinates())
            self.add(pr.Resize(shape=(self.size, self.size)))
            self.add(pr.CastImageToFloat())
            self.add(pr.SubtractMeanImage(self.mean))
            self.add(pr.MatchBoxes(prior_boxes, iou))
            self.add(pr.EncodeBoxes(prior_boxes, variances))
            self.add(pr.ToOneHotVector(num_classes))
            self.add(pr.OutputSelector(['image', 'boxes']))

        elif ((self.split == 'val') or (self.split == 'test')):
            self.add(pr.LoadImage())
            self.add(pr.CastImageToFloat())
            self.add(pr.Resize(self.size))
            self.add(pr.SubtractMeanImage(self.mean))

    @property
    def output_shapes(self):
        return [(self.size, self.size, 3),
                (len(self.prior_boxes), 4 + self.num_classes)]


class SingleShotInference(SequentialProcessor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN):
        super(SingleShotInference, self).__init__()

        self.add(pr.PredictBoxes(model, mean))
        self.add(pr.DecodeBoxes(model.prior_boxes, variances=[.1, .2]))
        self.add(pr.DetectBoxes(model.prior_boxes, nms_thresh=nms_thresh))
        self.add(pr.FilterBoxes(class_names, score_thresh))
        self.add(pr.DenormalizeBoxes2D())
        self.add(pr.DrawBoxes2D(class_names))
        self.add(pr.CastImageToInts())

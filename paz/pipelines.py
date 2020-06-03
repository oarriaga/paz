from .abstract import Processor, SequentialProcessor
from . import processors as pr


class AugmentImage(SequentialProcessor):
    def __init__(self):
        super(AugmentImage, self).__init__()
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())
        self.add(pr.RandomSaturation())
        self.add(pr.RandomHue())


class AugmentBoxes(SequentialProcessor):
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToAbsoluteBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class PreprocessImage(SequentialProcessor):
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        if mean is None:
            self.add(pr.NormalizeImage())
        else:
            self.add(pr.SubtractMeanImage(mean))


class PreprocessBoxes(SequentialProcessor):
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class AugmentDetection(SequentialProcessor):
    def __init__(self, prior_boxes, split=pr.TRAIN, num_classes=21, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5, variances=[.1, .2]):
        super(AugmentDetection, self).__init__()

        self.augment_image = AugmentImage()
        self.augment_image.insert(0, pr.LoadImage())
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(self.augment_image, [0], [0]))
        self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
            self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


class SingleShotInference(Processor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN):
        super(SingleShotInference, self).__init__()
        preprocessing = SequentialProcessor(
            [pr.ResizeImage(model.input_shape[1:3]),
             pr.ConvertColorSpace(pr.RGB2BGR),
             pr.SubtractMeanImage(mean),
             pr.CastImage(float),
             pr.ExpandDims(axis=0)])
        postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(model.prior_boxes, variances=[.1, .2]),
             pr.NonMaximumSuppressionPerClass(nms_thresh),
             pr.FilterBoxes(class_names, score_thresh)])
        self.predict = pr.Predict(model, preprocessing, postprocessing)

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw = pr.DrawBoxes2D(class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


class KeypointNetInference(Processor):
    def __init__(self, model, num_keypoints=None, radius=5):

        super(KeypointNetInference, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]

        self.predict_keypoints = SequentialProcessor()
        preprocessing = [pr.NormalizeImage(), pr.ExpandDims(axis=0)]
        self.predict_keypoints.add(pr.Predict(model, preprocessing))
        self.predict_keypoints.add(pr.SelectElement(0))
        self.predict_keypoints.add(pr.Squeeze(axis=0))
        self.predict_keypoints.add(pr.DenormalizeKeypoints())
        self.predict_keypoints.add(pr.RemoveKeypointsDepth())
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict_keypoints(image)
        image = self.draw(image, keypoints)
        return self.wrap(image, keypoints)


class RenderTwoViews(Processor):
    def __init__(self, renderer):
        super(RenderTwoViews, self).__init__()
        self.render = pr.Render(renderer)

        self.preprocess_image = SequentialProcessor()
        self.preprocess_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image.add(pr.NormalizeImage())

        self.preprocess_alpha = SequentialProcessor()
        self.preprocess_alpha.add(pr.ExpandDims(-1))
        self.preprocess_alpha.add(pr.NormalizeImage())
        self.concatenate = pr.Concatenate(-1)

    def call(self):
        image_A, image_B, matrices, alpha_A, alpha_B = self.render()
        image_A = self.preprocess_image(image_A)
        image_B = self.preprocess_image(image_B)
        alpha_A = self.preprocess_alpha(alpha_A)
        alpha_B = self.preprocess_alpha(alpha_B)
        alpha_channels = self.concatenate([alpha_A, alpha_B])
        return image_A, image_B, matrices, alpha_channels


class KeypointSharedAugmentation(SequentialProcessor):
    def __init__(self, renderer, size):
        super(KeypointSharedAugmentation, self).__init__()
        self.renderer = renderer
        self.size = size
        self.add(RenderTwoViews(self.renderer))
        self.add(pr.SequenceWrapper(
            {0: {'image_A': [size, size, 3]},
             1: {'image_B': [size, size, 3]}},
            {2: {'matrices': [4, 4 * 4]},
             3: {'alpha_channels': [size, size, 2]}})

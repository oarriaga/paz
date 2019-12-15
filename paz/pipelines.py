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
            self.add(pr.RandomSaturation())
            self.add(pr.RandomHue())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('HSV', to='BGR'))
            self.add(pr.RandomLightingNoise())
            self.add(pr.ToAbsoluteCoordinates())
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
            self.add(pr.OutputSelector(['image'], ['boxes']))

        elif ((self.split == 'val') or (self.split == 'test')):
            self.add(pr.LoadImage())
            self.add(pr.CastImageToFloat())
            self.add(pr.Resize((self.size, self.size)))
            self.add(pr.SubtractMeanImage(self.mean))
            self.add(pr.MatchBoxes(prior_boxes, iou))
            self.add(pr.EncodeBoxes(prior_boxes, variances))
            self.add(pr.ToOneHotVector(num_classes))
            self.add(pr.OutputSelector(['image'], ['boxes']))

    @property
    def input_shapes(self):
        return [(self.size, self.size, 3)]

    @property
    def label_shapes(self):
        return [(len(self.prior_boxes), 4 + self.num_classes)]


class SingleShotInference(SequentialProcessor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN):
        super(SingleShotInference, self).__init__()
        self.model, self.class_names = model, class_names
        self.score_thresh, self.nms_thresh = score_thresh, nms_thresh
        self.mean = mean

        self.add(pr.PredictBoxes(model, mean))
        self.add(pr.DecodeBoxes(model.prior_boxes, variances=[.1, .2]))
        self.add(pr.NonMaximumSuppressionPerClass(nms_thresh))
        self.add(pr.FilterBoxes(class_names, score_thresh))
        self.add(pr.DenormalizeBoxes2D())
        self.add(pr.DrawBoxes2D(class_names))
        self.add(pr.CastImageToInts())


class KeypointSharedAugmentation(SequentialProcessor):
    def __init__(self, renderer, projector, size):
        super(KeypointSharedAugmentation, self).__init__()
        self.renderer = renderer
        self.size = size
        self.add(pr.RenderMultiViewSample(self.renderer))
        self.add(pr.ConvertColor('RGB', to='BGR', topic='image_A'))
        self.add(pr.ConvertColor('RGB', to='BGR', topic='image_B'))
        self.add(pr.NormalizeImage('image_A'))
        self.add(pr.NormalizeImage('image_B'))
        self.add(pr.NormalizeImage('alpha_channels'))
        self.add(pr.OutputSelector(
            ['image_A', 'image_B'], ['matrices', 'alpha_channels']))

    @property
    def input_shapes(self):
        return [(self.size, self.size, 3),
                (self.size, self.size, 3)]

    @property
    def label_shapes(self):
        return [(4, 4 * 4), (self.size, self.size, 2)]


class KeypointAugmentation(SequentialProcessor):
    def __init__(self, renderer, projector, keypoints, split='train',
                 image_paths=None, size=128, with_partition=False,
                 num_occlusions=0, max_radius_scale=0.5,
                 plain_color=[0, 0, 0], with_geometric_transforms=False):

        super(KeypointAugmentation, self).__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split mode')

        self.renderer = renderer
        self.projector = projector
        self.keypoints = keypoints
        self.image_paths = image_paths
        self.split = split
        self.size = size
        self.with_partition = with_partition
        self.num_occlusions = num_occlusions
        self.max_radius_scale = max_radius_scale
        self.plain_color = plain_color
        self.with_geometric_transforms = with_geometric_transforms

        self.add(pr.RenderSingleViewSample(self.renderer))
        self.add(pr.ConvertColor('RGB', to='BGR'))
        self.add(pr.ProjectKeypoints(self.projector, self.keypoints))

        if split == 'train':
            self.add(pr.CastImageToFloat())
            self.add(pr.ConcatenateAlphaMask())
            if image_paths is not None:
                self.add(pr.AddCroppedBackground(image_paths, size))
            else:
                self.add(pr.AddPlainBackground(self.plain_color))

            for occlusion_arg in range(self.num_occlusions):
                self.add(pr.AddOcclusion(self.max_radius_scale))
            self.add(pr.CastImageToFloat())
            self.add(pr.RandomBlur())
            self.add(pr.RandomContrast())
            self.add(pr.RandomBrightness())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('BGR', to='HSV'))
            self.add(pr.CastImageToFloat())
            self.add(pr.RandomSaturation())
            self.add(pr.RandomHue())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('HSV', to='BGR'))
            # self.add(pr.RandomLightingNoise())
            if with_geometric_transforms:
                self.add(pr.DenormalizeKeypoints())
                self.add(pr.ApplyRandomTranslation())
                self.add(pr.Expand())
                self.add(pr.NormalizeKeypoints())

        self.add(pr.RemoveKeypointsDepth())
        self.add(pr.Resize(shape=(self.size, self.size)))
        self.add(pr.CastImageToFloat())
        self.add(pr.NormalizeImage())
        if self.with_partition:
            self.add(pr.PartitionKeypoints())
            num_keypoints = len(self.keypoints)
            label_topics = ['keypoint_%s' % i for i in range(num_keypoints)]
        else:
            label_topics = ['keypoints']
        self.add(pr.OutputSelector(['image'], label_topics))

    @property
    def input_shapes(self):
        return [(self.size, self.size, 3)]

    @property
    def label_shapes(self):
        if self.with_partition:
            return [(2,) for _ in range(len(self.keypoints))]
        else:
            return [(len(self.keypoints), 2)]


class KeypointNetInference(SequentialProcessor):
    """KeypointNet inference pipeline.
    # Arguments
        model: Keras model.
        num_keypoints: Int.
        radius: Int.
    # Returns
        Function for outputting keypoints from image
    """
    def __init__(self, model, num_keypoints=None, radius=5):

        super(KeypointNetInference, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]
        pipeline = [pr.NormalizeImage(), pr.ExpandDims(axis=0, topic='image')]
        self.add(pr.Predict(model, 'image', 'keypoints', pipeline))
        self.add(pr.SelectElement('keypoints', 0))
        self.add(pr.Squeeze(axis=0, topic='keypoints'))
        self.add(pr.DenormalizeKeypoints())
        self.add(pr.RemoveKeypointsDepth())
        self.add(pr.DrawKeypoints2D(self.num_keypoints, self.radius, False))
        self.add(pr.CastImageToInts())


class KeypointInference(SequentialProcessor):
    """General keypoint inference pipeline.
    # Arguments
        model: Keras model.
        num_keypoints: Int.
        radius: Int.
    # Returns
        Function for outputting keypoints from image
    """
    def __init__(self, model, num_keypoints=None, radius=5):
        super(KeypointInference, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]

        pipeline = [pr.Resize(model.input_shape[1:3]),
                    pr.NormalizeImage(),
                    pr.ExpandDims(axis=0, topic='image')]

        self.add(pr.Predict(model, 'image', 'keypoints', pipeline))
        self.add(pr.Squeeze(axis=0, topic='keypoints'))
        self.add(pr.DenormalizeKeypoints())
        self.add(pr.DrawKeypoints2D(self.num_keypoints, self.radius, False))
        self.add(pr.CastImageToInts())


class KeypointToPoseInference(SequentialProcessor):
    """General keypoint inference pipeline.
    # Arguments
        model: Keras model.
        num_keypoints: Int.
        radius: Int.
    # Returns
        Function for outputting pose from image
    """
    def __init__(self, model, points3D, camera, class_to_dimensions, radius=5):

        super(KeypointToPoseInference, self).__init__()
        self.num_keypoints = model.output_shape[1]
        self.radius = radius

        pipeline = [pr.Resize(model.input_shape[1:3]),
                    pr.NormalizeImage(),
                    pr.ExpandDims(axis=0, topic='image')]

        self.add(pr.Predict(model, 'image', 'keypoints', pipeline))
        self.add(pr.Squeeze(axis=0, topic='keypoints'))
        self.add(pr.DenormalizeKeypoints())
        self.add(pr.SolvePNP(points3D, camera))
        self.add(pr.DrawBoxes3D(camera, class_to_dimensions))
        self.add(pr.DrawKeypoints2D(self.num_keypoints, self.radius, False))
        self.add(pr.CastImageToInts())

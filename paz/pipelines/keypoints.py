from tensorflow.keras.utils import get_file

from .renderer import RenderTwoViews
from .image import PreprocessImageHigherHRNet
from .heatmaps import GetHeatmapsAndTags

from .. import processors as pr
from ..abstract import SequentialProcessor, Processor
from ..models import KeypointNet2D, HigherHRNet, DetNet

from ..backend.image import get_affine_transform, flip_left_right
from ..datasets import JOINT_CONFIG, FLIP_CONFIG


class KeypointNetSharedAugmentation(SequentialProcessor):
    """Wraps ``RenderTwoViews`` as a sequential processor for using it directly
        with a ``paz.GeneratingSequence``.

    # Arguments
        renderer: ``RenderTwoViews`` processor.
        size: Image size.
    """
    def __init__(self, renderer, size):
        super(KeypointNetSharedAugmentation, self).__init__()
        self.renderer = renderer
        self.size = size
        self.add(RenderTwoViews(self.renderer))
        self.add(pr.SequenceWrapper(
            {0: {'image_A': [size, size, 3]},
             1: {'image_B': [size, size, 3]}},
            {2: {'matrices': [4, 4 * 4]},
             3: {'alpha_channels': [size, size, 2]}}))


class KeypointNetInference(Processor):
    """Performs inference from a ``KeypointNetShared`` model.

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ``num_keypoints`` is
            tried to be inferred from ``model.output_shape``
        radius: Int. used for drawing the predicted keypoints.
    """
    def __init__(self, model, num_keypoints=None, radius=5):
        super(KeypointNetInference, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]

        preprocessing = SequentialProcessor()
        preprocessing.add(pr.NormalizeImage())
        preprocessing.add(pr.ExpandDims(axis=0))
        self.predict_keypoints = SequentialProcessor()
        self.predict_keypoints.add(pr.Predict(model, preprocessing))
        self.predict_keypoints.add(pr.SelectElement(0))
        self.predict_keypoints.add(pr.Squeeze(axis=0))
        self.postprocess_keypoints = SequentialProcessor()
        self.postprocess_keypoints.add(pr.DenormalizeKeypoints())
        self.postprocess_keypoints.add(pr.RemoveKeypointsDepth())
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict_keypoints(image)
        keypoints = self.postprocess_keypoints(keypoints, image)
        image = self.draw(image, keypoints)
        return self.wrap(image, keypoints)


class EstimateKeypoints2D(Processor):
    """Basic 2D keypoint prediction pipeline.

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ``num_keypoints`` is
            tried to be inferred from ``model.output_shape``
        draw: Boolean indicating if inferences should be drawn.
        radius: Int. used for drawing the predicted keypoints.
    """
    def __init__(self, model, num_keypoints, draw=True, radius=3,
                 color=pr.RGB2BGR):
        self.model = model
        self.num_keypoints = num_keypoints
        self.draw, self.radius, self.color = draw, radius, color
        self.preprocess = SequentialProcessor()
        self.preprocess.add(pr.ResizeImage(self.model.input_shape[1:3]))
        self.preprocess.add(pr.ConvertColorSpace(self.color))
        self.preprocess.add(pr.NormalizeImage())
        self.preprocess.add(pr.ExpandDims(0))
        self.preprocess.add(pr.ExpandDims(-1))
        self.predict = pr.Predict(model, self.preprocess, pr.Squeeze(0))
        self.denormalize = pr.DenormalizeKeypoints()
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict(image)
        keypoints = self.denormalize(keypoints, image)
        if self.draw:
            image = self.draw(image, keypoints)
        return self.wrap(image, keypoints)


class FaceKeypointNet2D32(EstimateKeypoints2D):
    """KeypointNet2D model trained with Kaggle Facial Detection challenge.

    # Arguments
        draw: Boolean indicating if inferences should be drawn.
        radius: Int. used for drawing the predicted keypoints.

    # Example
        ``` python
        from paz.pipelines import FaceKeypointNet2D32

        estimate_keypoints= FaceKeypointNet2D32()

        # apply directly to an image (numpy-array)
        inference = estimate_keypoints(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``keypoints``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a numpy array representing the keypoints.
    """
    def __init__(self, draw=True, radius=3):
        model = KeypointNet2D((96, 96, 1), 15, 32, 0.1)
        self.weights_URL = ('https://github.com/oarriaga/altamira-data/'
                            'releases/download/v0.7/')
        weights_path = self.get_weights_path(model)
        model.load_weights(weights_path)
        super(FaceKeypointNet2D32, self).__init__(
            model, 15, draw, radius, pr.RGB2GRAY)

    def get_weights_path(self, model):
        model_name = '_'.join(['FaceKP', model.name, '32', '15'])
        model_name = '%s_weights.hdf5' % model_name
        URL = self.weights_URL + model_name
        return get_file(model_name, URL, cache_subdir='paz/models')


class GetKeypoints(Processor):
    """Extract out the top k keypoints heatmaps and group the keypoints with
       their respective tags value. Adjust and refine the keypoint locations
       by removing the margins.
    # Arguments
        max_num_instance: Int. Maximum number of instances to be detected.
        keypoint_order: List of length 17 (number of keypoints).
        heatmaps: Numpy array of shape (1, num_keypoints, H, W)
        Tags: Numpy array of shape (1, num_keypoints, H, W, 2)

    # Returns
        grouped_keypoints: numpy array. keypoints grouped by tag
        scores: int: score for the keypoint
    """
    def __init__(self, max_num_instance, keypoint_order, detection_thresh=0.2,
                 tag_thresh=1):
        super(GetKeypoints, self).__init__()
        self.group_keypoints = pr.SequentialProcessor(
            [pr.TopKDetections(max_num_instance), pr.GroupKeypointsByTag(
                keypoint_order, tag_thresh, detection_thresh)])
        self.adjust_keypoints = pr.AdjustKeypointsLocations()
        self.get_scores = pr.GetScores()
        self.refine_keypoints = pr.RefineKeypointsLocations()

    def call(self, heatmaps, tags, adjust=True, refine=True):
        grouped_keypoints = self.group_keypoints(heatmaps, tags)
        if adjust:
            grouped_keypoints = self.adjust_keypoints(
                heatmaps, grouped_keypoints)[0]
        scores = self.get_scores(grouped_keypoints)
        if refine:
            grouped_keypoints = self.refine_keypoints(
                heatmaps[0], tags[0], grouped_keypoints)
        return grouped_keypoints, scores


class TransformKeypoints(Processor):
    """Transform the keypoint coordinates.
    # Arguments
        grouped_keypoints: Numpy array. keypoints grouped by tag
        center: Tuple. center of the imput image
        scale: Float. scaled imput image dimension
        shape: Tuple/List

    # Returns
        transformed_keypoints: keypoint location with respect to the
                               input image
    """
    def __init__(self, inverse=False):
        super(TransformKeypoints, self).__init__()
        self.inverse = inverse
        self.get_source_destination_point = pr.GetSourceDestinationPoints(
            scaling_factor=200)
        self.transform_keypoints = pr.TransformKeypoints()

    def call(self, grouped_keypoints, center, scale, shape):
        source_point, destination_point = self.get_source_destination_point(
            center, scale, shape)
        if self.inverse:
            source_point, destination_point = destination_point, source_point
        transform = get_affine_transform(source_point, destination_point)
        transformed_keypoints = self.transform_keypoints(grouped_keypoints,
                                                         transform)
        return transformed_keypoints


class HigherHRNetHumanPose2D(Processor):
    """Estimate human pose 2D keypoints and draw a skeleton.

    # Arguments
        model: Weights trained on HigherHRNet model.
        keypoint_order: List of length 17 (number of keypoints).
            where the keypoints are listed order wise.
        flipped_keypoint_order: List of length 17 (number of keypoints).
            Flipped list of keypoint order.
        dataset: String. Name of the dataset used for training the model.
        data_with_center: Boolean. True is the model is trained using the
            center.

    # Returns
        dictonary with the following keys:
            image: contains the image with skeleton drawn on it.
            keypoints: location of keypoints
            score: score of detection
    """
    def __init__(self, dataset='COCO', data_with_center=False,
                 max_num_people=30, with_flip=True, draw=True):
        super(HigherHRNetHumanPose2D, self).__init__()
        keypoint_order = JOINT_CONFIG[dataset]
        flipped_keypoint_order = FLIP_CONFIG[dataset]
        self.with_flip = with_flip
        self.draw = draw
        self.model = HigherHRNet(weights=dataset)
        self.transform_image = PreprocessImageHigherHRNet()
        self.get_heatmaps_and_tags = pr.SequentialProcessor(
            [GetHeatmapsAndTags(self.model, flipped_keypoint_order,
             with_flip, data_with_center), pr.AggregateResults(with_flip)])
        self.get_keypoints = GetKeypoints(max_num_people, keypoint_order)
        self.transform_keypoints = TransformKeypoints(inverse=True)
        self.draw_skeleton = pr.DrawHumanSkeleton(dataset, check_scores=True)
        self.extract_keypoints_locations = pr.ExtractKeypointsLocations()
        self.wrap = pr.WrapOutput(['image', 'keypoints', 'scores'])

    def call(self, image):
        resized_image, center, scale = self.transform_image(image)
        heatmaps, tags = self.get_heatmaps_and_tags(resized_image)
        keypoints, scores = self.get_keypoints(heatmaps, tags)
        shape = [heatmaps.shape[3], heatmaps.shape[2]]
        keypoints = self.transform_keypoints(keypoints, center, scale, shape)
        if self.draw:
            image = self.draw_skeleton(image, keypoints)
        keypoints = self.extract_keypoints_locations(keypoints)
        return self.wrap(image, keypoints, scores)


class HandPoseEstimation(Processor):
    """Hand keypoints detection pipeline.

    # Arguments
        hand_estimator: Keras model for predicting keypoints.
        shape: Tuple. Shape the input image to be reshaped. eg (128, 128)
        draw: Boolean indicating if inferences should be drawn.
    """
    def __init__(self, hand_estimator, shape=(128, 128), draw=True):
        super(HandPoseEstimation).__init__()
        self.draw = draw
        self.preprocess = SequentialProcessor(
            [pr.ResizeImage(shape), pr.ExpandDims(axis=0)])
        self.hand_estimator = hand_estimator
        self.scale_keypoints = pr.ScaleKeypoints(scale=4, shape=shape)
        self.draw_skeleton = pr.DrawHandSkeleton()
        self.wrap = pr.WrapOutput(['image', 'keypoints3D', 'keypoints2D'])

    def call(self, input_image):
        image = self.preprocess(input_image)
        keypoints3D, keypoints2D = self.hand_estimator.predict(image)
        keypoints2D = flip_left_right(keypoints2D)
        keypoints2D = self.scale_keypoints(keypoints2D, input_image)
        if self.draw:
            image = self.draw_skeleton(input_image, keypoints2D)
        return self.wrap(image, keypoints3D, keypoints2D)


class MinimalHandPoseEstimation(HandPoseEstimation):
    """
        Minimal hand keypoints detection using DetNet model.
    """
    def __init__(self):
        detect_hand = DetNet()
        super(MinimalHandPoseEstimation, self).__init__(detect_hand)

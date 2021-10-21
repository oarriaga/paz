from paz import processors as pr

from paz.abstract import SequentialProcessor, Processor
from processors import AdjustCropSize, CropImage, CanonicaltoRelativeFrame
from processors import CreateScoremaps, ExtractBoundingbox
from processors import Extract2DKeypoints, ExtractHandSide, FlipRightHand
from processors import ExtractDominantKeypoint, CropImageFromMask
from processors import ExtractHandmask, KeypointstoPalmFrame
from processors import MatrixInverse, ExtractDominantHandVisibility
from processors import Resize_image, RotationMatrixfromAxisAngles
from processors import TransformVisibilityMask, NormalizeKeypoints
from processors import TransformtoRelativeFrame, GetCanonicalTransformation
from processors import ToOneHot
from layer import SegmentationDilation


class PreprocessKeypoints(SequentialProcessor):
    def __init__(self):
        super(PreprocessKeypoints, self).__init__()
        self.add(ExtractHandmask())  # use it directly in main pipeline


class AugmentHandSegmentation(SequentialProcessor):
    def __init__(self, size=320):
        super(AugmentHandSegmentation, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'segmentation_label',
                                      'annotations']))

        preprocess_image = pr.SequentialProcessor(
            [pr.LoadImage(),
             pr.ResizeImage((size, size))])

        preprocess_segmentation_map = pr.SequentialProcessor(
            [pr.LoadImage(),
             pr.ResizeImage((size, size)),
             ExtractHandmask()])

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_segmentation_map, [1], [1]))

        self.add(pr.SequenceWrapper({0: {'image': [size, size, 3]}},
                                    {1: {'hand_mask': [size, size]}}))


class AugmentHandPose2D(Processor):
    def __init__(self, size, image_size, crop_size, variance):
        super(AugmentHandPose2D, self).__init__()
        self.unwrap_inputs = pr.UnpackDictionary(
            ['image', 'segmentation_label', 'annotations'])
        self.preprocess_image = pr.SequentialProcessor(
            [pr.LoadImage(),
             pr.ResizeImage((size, size))])

        self.preprocess_segmentation_map = pr.SequentialProcessor(
            [pr.LoadImage(),
             pr.ResizeImage((size, size)),
             ExtractHandmask()])
        self.extract_annotations = pr.UnpackDictionary(['xyz', 'uv_vis', 'K'])
        self.extract_2D_keypoints = Extract2DKeypoints()
        self.keypoints_to_palm = KeypointstoPalmFrame()
        self.visibility_to_palm = TransformVisibilityMask()
        self.extract_hand_side = ExtractHandSide()

        self.extract_visibility_dominant_hand = ExtractDominantHandVisibility()
        self.create_scoremaps = CreateScoremaps(image_size=image_size,
                                                crop_size=crop_size,
                                                variance=variance)
        self.crop_image_from_mask = CropImageFromMask()
        self.wrap = pr.WrapOutput(
            ['cropped_image', 'score_maps', 'keypoints_vis21'])

    def call(self, inputs, use_palm_coordinates, crop_image):
        image, segmentation_label, annotations = self.unwrap_inputs(inputs)

        image = self.preprocess_image(image)
        segmentation_label = self.preprocess_segmentation_map(
            segmentation_label)
        keypoints3D, keypoints2D, camera_matrix = self.extract_annotations(
            annotations)
        keypoints2D, keypoints_visibility_mask = self.extract_2D_keypoints(
            keypoints2D)

        if use_palm_coordinates:
            keypoints2D = self.keypoints_to_palm(keypoints2D)
            keypoints_visibility_mask = self.visibility_to_palm(
                keypoints_visibility_mask)

        hand_side, keypoints3D, dominant_hand = self.extract_hand_side(
            segmentation_label, keypoints3D)

        keypoints21 = self.extract_visibility_dominant_hand(
            keypoints_visibility_mask, dominant_hand)

        scoremaps = self.create_scoremaps(keypoints2D, keypoints21)

        if crop_image:
            image = self.crop_image_from_mask(
                keypoints2D, keypoints21, image, camera_matrix)

        return self.wrap(image, scoremaps, keypoints21)


class AugmentHandPose(Processor):
    def __init__(self, size, image_size, crop_size, variance):
        super(AugmentHandPose, self).__init__()
        self.unwrap_inputs = pr.UnpackDictionary(
            ['image', 'segmentation_label', 'annotations'])
        self.preprocess_image = pr.SequentialProcessor(
            [pr.LoadImage(),
             pr.ResizeImage((size, size))])

        self.preprocess_segmentation_map = pr.SequentialProcessor(
            [pr.LoadImage(),
             pr.ResizeImage((size, size)),
             ExtractHandmask()])

        self.extract_annotations = pr.UnpackDictionary(['xyz', 'uv_vis', 'K'])
        self.extract_2D_keypoints = Extract2DKeypoints()
        self.keypoints_to_palm = KeypointstoPalmFrame()
        self.visibility_to_palm = TransformVisibilityMask()
        self.extract_hand_side = ExtractHandSide()
        self.to_one_hot = ToOneHot(num_classes=2)
        self.normaliza_keypoints = NormalizeKeypoints()
        self.to_relative_frame = TransformtoRelativeFrame()
        self.canonical_transformations = GetCanonicalTransformation()
        self.flip_right_hand = FlipRightHand()
        self.get_matrix_inverse = MatrixInverse()

        self.extract_hand_visibility = ExtractDominantHandVisibility()
        self.extract_dominant_keypoints = ExtractDominantKeypoint()

        self.crop_image_from_mask = CropImageFromMask()
        self.create_scoremaps = CreateScoremaps(image_size=image_size,
                                                crop_size=crop_size,
                                                variance=variance)

        self.wrap = pr.WrapOutput(
            ['score_maps', 'hand_side', 'keypoints3D', 'rotation_matrix'])

    def call(self, inputs, use_palm_coordinates, crop_image,
             flip_right_hand=False):
        image, segmentation_label, annotations = self.unwrap_inputs(inputs)

        image = self.preprocess_image(image)
        segmentation_label = self.preprocess_segmentation_map(
            segmentation_label)
        keypoints3D, keypoints2D, camera_matrix = self.extract_annotations(
            annotations)
        keypoints2D, keypoints_visibility_mask = self.extract_2D_keypoints(
            keypoints2D)

        if use_palm_coordinates:
            keypoints2D = self.keypoints_to_palm(keypoints2D)
            keypoints3D = self.keypoints_to_palm(keypoints3D)
            keypoints_visibility_mask = self.visibility_to_palm(
                keypoints_visibility_mask)

        hand_side, keypoints3D, dominant_hand = self.extract_hand_side(
            segmentation_label, keypoints3D)

        hand_side_one_hot = self.to_one_hot(hand_side)

        keypoint_scale, keypoints3D = self.normaliza_keypoints(keypoints3D)
        keypoints3D = self.to_relative_frame(keypoints3D)
        keypoints3D, canonical_rotation_matrix = self.canonical_transformations(
            keypoints3D)

        if flip_right_hand:
            keypoints3D = self.flip_right_hand(keypoints3D)

        canonical_rotation_matrix = self.get_matrix_inverse(
            canonical_rotation_matrix)

        visible_keypoints = self.extract_hand_visibility(
            keypoints_visibility_mask, dominant_hand)
        dominant_keypoints = self.extract_dominant_keypoints(
            keypoints2D, dominant_hand)

        if crop_image:
            scale, image, visible_keypoints, camera_matrix = \
                self.crop_image_from_mask(
                    visible_keypoints, dominant_keypoints, image, camera_matrix)
        scoremaps = self.create_scoremaps(
            canonical_rotation_matrix, visible_keypoints)

        return self.wrap(scoremaps, hand_side_one_hot, keypoints3D,
                         canonical_rotation_matrix)


class PreprocessImage(SequentialProcessor):
    def __init__(self, image_size=320):
        super(PreprocessImage, self).__init__()
        self.add(pr.NormalizeImage())
        self.add(pr.ResizeImage((image_size, image_size)))
        self.add(pr.ExpandDims(0))


class PostprocessSegmentation(SequentialProcessor):
    def __init__(self, HandSegNet, image_size=320, crop_size=256):
        super(PostprocessSegmentation, self).__init__()
        self.add(pr.Predict(HandSegNet))
        self.add(pr.UnpackDictionary(['image', 'raw_segmentation_map']))
        self.add(pr.ControlMap(Resize_image(size=(image_size, image_size)),
                               [1], [1]))
        self.add(pr.ControlMap(SegmentationDilation(), [1], [1]))
        self.add(pr.ControlMap(ExtractBoundingbox(), [1], [2, 3, 4],
                               keep={1: 1}))
        self.add(pr.ControlMap(AdjustCropSize(), [4], [4]))
        self.add(pr.ControlMap(CropImage(crop_size=crop_size), [0, 2, 4],
                               [0], keep={2: 2, 4: 4}))


class Process2DKeypoints(SequentialProcessor):
    def __init__(self, PoseNet):
        super(Process2DKeypoints, self).__init__()
        self.add(pr.ExpandDims(0))
        self.add(pr.Predict(PoseNet))


class PostProcessKeypoints(SequentialProcessor):
    def __init__(self, number_of_keypoints=21):
        super(PostProcessKeypoints, self).__init__()
        self.add(pr.UnpackDictionary(['canonical_coordinates',
                                      'rotation_parameters', 'hand_side']))
        self.add(pr.ControlMap(RotationMatrixfromAxisAngles(), [1], [1]))
        self.add(pr.ControlMap(CanonicaltoRelativeFrame(number_of_keypoints),
                               [0, 1, 2], [0]))

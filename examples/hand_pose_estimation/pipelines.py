import numpy as np

from layer import SegmentationDilation
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor, Box2D
from processors_SE3 import CalculatePseudoInverse, RotationMatrixfromAxisAngles
from processors_SE3 import CanonicaltoRelativeFrame, KeypointstoPalmFrame
from processors_SE3 import GetCanonicalTransformation, TransformKeypoints
from processors_SE3 import TransformVisibilityMask, TransformtoRelativeFrame
from processors_keypoints import AdjustCropSize, CropImage
from processors_keypoints import CreateScoremaps, ExtractBoundingbox
from processors_keypoints import Extract2DKeypoints, ExtractHandsideandKeypoints
from processors_keypoints import ExtractDominantHandVisibility
from processors_keypoints import ExtractDominantKeypoints2D, CropImageFromMask
from processors_keypoints import ExtractHandmask, ExtractKeypoints
from processors_keypoints import FlipRightHandToLeftHand
from processors_keypoints import NormalizeKeypoints
from processors_standard import MergeDictionaries, ToOneHot, WrapToDictionary
from processors_standard import ResizeImageWithLinearInterpolation
from processors_standard import TransposeOfArray, ListToArray


class ExtractHandSegmentation(SequentialProcessor):
    def __init__(self, size=320):
        super(ExtractHandSegmentation, self).__init__()
        self.add(pr.UnpackDictionary(
            ['image', 'segmentation_label', 'annotations']))

        preprocess_image = pr.SequentialProcessor(
            [pr.LoadImage(), pr.ResizeImage((size, size))])

        preprocess_segmentation_map = pr.SequentialProcessor(
            [pr.LoadImage(), pr.ResizeImage((size, size)), ExtractHandmask()])

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_segmentation_map, [1], [1]))
        self.add(pr.SequenceWrapper({0: {'image': [size, size, 3]}},
                                    {1: {'hand_mask': [size, size]}}))


class ExtractHandPose2D(Processor):
    def __init__(self, size, image_size, crop_size, variance):
        super(ExtractHandPose2D, self).__init__()
        self.unwrap_inputs = pr.UnpackDictionary(
            ['image', 'segmentation_label', 'annotations'])
        self.preprocess_image = pr.SequentialProcessor(
            [pr.LoadImage(), pr.ResizeImage((size, size))])

        self.preprocess_segmentation_map = pr.SequentialProcessor(
            [pr.LoadImage(), pr.ResizeImage((size, size)), ExtractHandmask()])
        self.extract_annotations = pr.UnpackDictionary(['xyz', 'uv_vis', 'K'])
        self.extract_2D_keypoints = Extract2DKeypoints()
        self.keypoints_to_palm = KeypointstoPalmFrame()
        self.visibility_to_palm = TransformVisibilityMask()
        self.extract_hand_side = ExtractHandsideandKeypoints()

        self.extract_visibility_dominant_hand = ExtractDominantHandVisibility()
        self.create_scoremaps = CreateScoremaps(
            image_size, crop_size, variance)
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


class ExtractHandPose(Processor):
    def __init__(self, size, image_size, crop_size, variance):
        super(ExtractHandPose, self).__init__()
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
        self.extract_hand_side = ExtractHandsideandKeypoints()
        self.to_one_hot = ToOneHot(num_classes=2)
        self.normaliza_keypoints = NormalizeKeypoints()
        self.to_relative_frame = TransformtoRelativeFrame()
        self.canonical_transformations = GetCanonicalTransformation()
        self.flip_right_hand = FlipRightHandToLeftHand()
        self.get_matrix_inverse = CalculatePseudoInverse()

        self.extract_hand_visibility = ExtractDominantHandVisibility()
        self.extract_dominant_keypoints = ExtractDominantKeypoints2D()

        self.crop_image_from_mask = CropImageFromMask()
        self.create_scoremaps = CreateScoremaps(
            image_size=image_size, crop_size=crop_size, variance=variance)

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


class PostProcessSegmentation(Processor):
    def __init__(self, image_size=320, crop_shape=(256, 256)):
        super(PostProcessSegmentation, self).__init__()
        self.unpack_inputs = pr.UnpackDictionary(['image',
                                                  'raw_segmentation_map'])
        self.resize_segmentation_map = ResizeImageWithLinearInterpolation(
            shape=(image_size, image_size))
        self.dilate_map = SegmentationDilation()
        self.extract_box = ExtractBoundingbox()
        self.adjust_crop_size = AdjustCropSize()
        self.crop_image = CropImage(crop_shape[0])
        self.expand_dims = pr.ExpandDims(axis=0)
        self.squeeze_input = pr.Squeeze(axis=0)

    def call(self, inputs):
        image, raw_segmentation_map = self.unpack_inputs(inputs)
        raw_segmentation_map = self.squeeze_input(raw_segmentation_map)
        raw_segmentation_map = self.resize_segmentation_map(
            raw_segmentation_map)
        segmentation_map = self.dilate_map(raw_segmentation_map)
        if not np.count_nonzero(segmentation_map):
            return None
        center, bounding_box, crop_size = self.extract_box(segmentation_map)
        crop_size = self.adjust_crop_size(crop_size)
        cropped_image = self.crop_image(image, center, crop_size)
        return cropped_image, segmentation_map, center, bounding_box, crop_size


class ResizeScoreMaps(Processor):  # Change to Sequential processor
    def __init__(self, crop_shape=(256, 256)):
        super(ResizeScoreMaps, self).__init__()
        self.unpack_inputs = pr.UnpackDictionary(['score_maps'])
        self.crop_shape = crop_shape
        self.squeeze = pr.Squeeze(axis=0)
        self.transpose = TransposeOfArray()
        self.resize_scoremap = pr.ResizeImages(crop_shape)
        self.list_to_array = ListToArray()
        self.expand_dims = pr.ExpandDims(axis=0)

    def call(self, input):
        scoremaps = self.unpack_inputs(input)
        scoremaps = self.squeeze(scoremaps)
        scoremaps_transposed = self.transpose(scoremaps)
        scoremaps_resized = self.resize_scoremap(scoremaps_transposed)
        scoremaps_resized = self.list_to_array(scoremaps_resized)
        scoremaps_transposed = self.transpose(scoremaps_resized)
        return scoremaps_transposed


class DetectHandKeypoints(Processor):
    def __init__(self, handsegnet, posenet, posepriornet, viewpointnet,
                 image_size=320, crop_shape=(256, 256), num_keypoints=21):
        super(DetectHandKeypoints, self).__init__()

        self.preprocess_image = SequentialProcessor(
            [pr.NormalizeImage(), pr.ResizeImage((image_size, image_size)),
             pr.ExpandDims(0)])
        postprocess_segmentation = PostProcessSegmentation(image_size,
                                                           crop_shape)
        self.localize_hand = pr.Predict(handsegnet,
                                        postprocess=postprocess_segmentation)

        self.resize_scoremaps = ResizeScoreMaps(crop_shape)
        self.merge_dictionaries = MergeDictionaries()
        self.wrap_input = WrapToDictionary(['hand_side'])

        self.predict_keypoints2D = pr.Predict(posenet)
        self.predict_keypoints3D = pr.Predict(posepriornet)
        self.predict_keypoints_angles = pr.Predict(viewpointnet)
        self.postprocess_keypoints = PostProcessKeypoints()
        self.resize = pr.ResizeImage(shape=crop_shape)
        self.extract_2D_keypoints = ExtractKeypoints()
        self.transform_keypoints = TransformKeypoints()
        self.draw_keypoint = pr.DrawKeypoints2D(num_keypoints, normalized=True,
                                                radius=4)
        self.denormalize = pr.DenormalizeImage()
        self.wrap = pr.WrapOutput(['image', 'keypoints2D', 'keypoints3D'])
        self.expand_dims = pr.ExpandDims(axis=0)
        self.draw_boxes = pr.DrawBoxes2D(['hand'], [[0, 1, 0]])

    def call(self, input_image, hand_side=np.array([[1.0, 0.0]])):
        image = self.preprocess_image(input_image)
        hand_features = self.localize_hand(image)
        if hand_features is None:
            output = self.wrap(input_image.astype('uint8'), None, None)
            return output
        hand_crop, segmentation_map, center, box, crop_size_best = hand_features
        box = Box2D(box, score=1.0, class_name='hand')
        image = self.draw_boxes(np.squeeze(image), [box])
        hand_crop = self.expand_dims(hand_crop)
        score_maps = self.predict_keypoints2D(hand_crop)
        score_maps_resized = self.resize_scoremaps(score_maps)
        hand_side = {'hand_side': hand_side}
        score_maps = self.merge_dictionaries([score_maps, hand_side])
        keypoints_2D = self.extract_2D_keypoints(score_maps_resized)
        rotation_parameters = self.predict_keypoints3D(score_maps)
        viewpoints = self.predict_keypoints_angles(score_maps)
        canonical_keypoints = self.merge_dictionaries([rotation_parameters,
                                                       viewpoints])
        keypoints3D = self.postprocess_keypoints(canonical_keypoints)
        keypoints2D = self.transform_keypoints(keypoints_2D, center,
                                               crop_size_best, 256)
        image = self.draw_keypoint(np.squeeze(image), keypoints2D)
        image = self.denormalize(image)
        output = self.wrap(image.astype('uint8'), keypoints2D, keypoints3D)
        return output

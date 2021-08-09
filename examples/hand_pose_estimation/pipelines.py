from paz import processors as pr
from paz.abstract import SequentialProcessor
from processors import AdjustCropSize, CropImage, CanonicaltoRelativeFrame
from processors import CreateScoremaps, HandSegmentationMap, ExtractBoundingbox
from processors import Extract2DKeypoints, ExtractHandSide, FlipRightHand
from processors import ExtractDominantKeypoint, CropImageFromMask
from processors import ExtractHandmask, KeypointsWristFrame
from processors import MatrixInverse, ExtractDominantHandVisibility
from processors import TransformVisibilityMask, NormalizeKeypoints
from processors import TransformtoRelativeFrame, GetCanonicalTransformation
from processors import Resize_image


class PreprocessKeypoints(SequentialProcessor):
    def __init__(self):
        super(PreprocessKeypoints, self).__init__()
        self.add(ExtractHandmask())


class AugmentHandSegmentation(SequentialProcessor):
    def __init__(self, size=320):
        super(AugmentHandSegmentation, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'segmentation_label',
                                      'annotations']))

        preprocess_image = pr.SequentialProcessor()
        preprocess_image.add(pr.LoadImage())
        preprocess_image.add(pr.ResizeImage([size, size]))

        preprocess_segmentation_map = pr.SequentialProcessor()
        preprocess_segmentation_map.add(pr.LoadImage())
        preprocess_segmentation_map.add(pr.ResizeImage([size, size]))
        preprocess_segmentation_map.add(ExtractHandmask())

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_segmentation_map, [1], [1]))

        self.add(pr.SequenceWrapper({0: {'image': [size, size, 3]}},
                                    {1: {'hand_mask': [size, size]}}))


class AugmentHandPose2D(SequentialProcessor):
    def __init__(self, size=320, crop_size=256, use_palm_coordinates=False,
                 crop_image=True):
        super(AugmentHandPose2D, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'segmentation_label',
                                      'annotations']))

        preprocess_image = pr.SequentialProcessor()
        preprocess_image.add(pr.LoadImage())
        preprocess_image.add(pr.ResizeImage([size, size]))

        preprocess_segmentation_map = pr.SequentialProcessor()
        preprocess_segmentation_map.add(pr.LoadImage())
        preprocess_segmentation_map.add(pr.ResizeImage([size, size]))
        preprocess_segmentation_map.add(ExtractHandmask())

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_segmentation_map, [1], [1]))

        self.add(pr.ControlMap(pr.UnpackDictionary(['xyz', 'uv_vis', 'K']),
                               [2], [2, 3, 4]))
        self.add(pr.ControlMap(Extract2DKeypoints(), [3], [3, 5]))

        if not use_palm_coordinates:
            self.add(pr.ControlMap(KeypointsWristFrame, [3], [3]))
            self.add(pr.ControlMap(TransformVisibilityMask, [5], [5]))

        self.add(pr.ControlMap(ExtractHandSide, [1, 2], [6, 2, 7]))

        self.add(pr.ControlMap(ExtractDominantHandVisibility, [6, 7], [8]))

        self.add(pr.ControlMap(CreateScoremaps, [3, 8], [9]))

        if crop_image:
            self.add(pr.ControlMap(CropImageFromMask(), [3, 8, 0, 4],
                                   [10, 0, 3, 4]))

        self.add(pr.SequenceWrapper({0: {'cropped_image': [crop_size, crop_size,
                                                           3]}},
                                    {9: {'score_maps': [size, size]},
                                     8: {'keypoints_vis21': [size, size]}}))


class AugmentHandPose(SequentialProcessor):
    def __init__(self, size=320, crop_size=256, use_palm_coordinates=False,
                 flip_right_hand=False, crop_image=True):
        super(AugmentHandPose, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'segmentation_label',
                                      'annotations']))

        preprocess_image = pr.SequentialProcessor()
        preprocess_image.add(pr.LoadImage())
        preprocess_image.add(pr.ResizeImage([size, size]))

        preprocess_segmentation_map = pr.SequentialProcessor()
        preprocess_segmentation_map.add(pr.LoadImage())
        preprocess_segmentation_map.add(pr.ResizeImage([size, size]))
        preprocess_segmentation_map.add(ExtractHandmask())

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_segmentation_map, [1], [1]))
        self.add(pr.ControlMap(pr.UnpackDictionary(['xyz', 'uv_vis', 'K']),
                               [2], [2, 3, 4]))
        self.add(pr.ControlMap(Extract2DKeypoints, [3], [5, 6]))

        if not use_palm_coordinates:
            self.add(pr.ControlMap(KeypointsWristFrame, [2], [2]))
            self.add(pr.ControlMap(KeypointsWristFrame, [5], [5]))
            self.add(pr.ControlMap(TransformVisibilityMask, [6], [6]))

        self.add(pr.ControlMap(ExtractHandSide, [1, 2], [7, 2, 8]))
        self.add(pr.ControlMap(pr.BoxClassToOneHotVector(num_classes=2),
                               [7], [7]))

        self.add(pr.ControlMap(NormalizeKeypoints, [2], [9, 2]))
        self.add(pr.ControlMap(TransformtoRelativeFrame, [2], [2]))
        self.add(pr.ControlMap(GetCanonicalTransformation, [2], [2, 10]))

        if flip_right_hand:
            self.add(pr.ControlMap(FlipRightHand, [2], [2]))

        self.add(pr.ControlMap(MatrixInverse, [10], [10]))

        self.add(pr.ControlMap(ExtractDominantHandVisibility, [6, 8], [11]))
        self.add(pr.ControlMap(ExtractDominantKeypoint, [3, 8], [12]))

        if crop_image:
            self.add(pr.ControlMap(CropImageFromMask(), [11, 12, 0, 4],
                                   [12, 13, 11, 14]))
        self.add(pr.ControlMap(CreateScoremaps, [10, 11], [15]))

        self.add(pr.SequenceWrapper({0: {'score_maps': [crop_size, crop_size]},
                                     7: {'hand_side': [2]}},
                                    {2: {'keypoints3D_can': [21, 3]},
                                     10: {'rotation_matrix': [3, 3]}}))

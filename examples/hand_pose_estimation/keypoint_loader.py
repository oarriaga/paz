from paz.abstract import SequentialProcessor, Processor
from paz.pipelines.image import AugmentImage
from paz import processors as pr
from processors import ExtractHandmask, KeypointsWristFrame
from processors import Extract2DKeypoints, ExtractHandSide, FlipRightHand
from processors import TransformVisibilityMask, NormalizeKeypoints
from processors import TransformtoRelativeFrame, GetCanonicalTransformation
from processors import MatrixInverse, ExtractDominantHandVisibility
from processors import ExtractDominantKeypoint, CropImageFromMask
from processors import CreateScoremaps


class PreprocessKeypoints(SequentialProcessor):
    def __init__(self):
        super(PreprocessKeypoints, self).__init__()
        self.add(ExtractHandmask())


class Preprocessposeinput(SequentialProcessor):
    def __init__(self, size=320, use_palm_coordinates=False,
                 flip_right_hand=False, crop_image=True,
                 image_size=[320, 320, 3], crop_size=256, sigma=25.0):
        super(Preprocessposeinput, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'segmentation_label',
                                      'annotations']))

        preprocess_image = pr.SequentialProcessor()
        preprocess_image.add(pr.LoadImage())
        preprocess_image.add(pr.ResizeImage([size, size]))

        preprocess_segmentation_map = pr.SequentialProcessor()
        preprocess_segmentation_map.add(pr.LoadImage())
        preprocess_segmentation_map.add(ExtractHandmask())

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_segmentation_map, [1], [1]))
        self.add(pr.ControlMap(pr.UnpackDictionary(['xyz', 'uv_vis', 'K']),
                               [2], [2, 3, 4]))
        self.add(pr.ControlMap(Extract2DKeypoints, [3], [5, 6]))

        if not use_palm_coordinates:
            self.add(pr.ControlMap(KeypointsWristFrame(), [2], [2]))
            self.add(pr.ControlMap(KeypointsWristFrame(), [5], [5]))
            self.add(pr.ControlMap(TransformVisibilityMask(), [6], [6]))

        self.add(pr.ControlMap(ExtractHandSide(), [1, 2], [7, 2, 8]))
        self.add(pr.ControlMap(pr.BoxClassToOneHotVector(num_classes=2),
                               [7], [7]))

        self.add(pr.ControlMap(NormalizeKeypoints(), [2], [2]))
        self.add(pr.ControlMap(TransformtoRelativeFrame(), [2], [2]))
        self.add(pr.ControlMap(GetCanonicalTransformation(), [2], [2, 9]))

        if flip_right_hand:
            self.add(pr.ControlMap(FlipRightHand(), [2], [2]))

        self.add(pr.ControlMap(MatrixInverse(), [9], [9]))

        self.add(pr.ControlMap(ExtractDominantHandVisibility(), [6, 8], [10]))
        self.add(pr.ControlMap(ExtractDominantKeypoint(), [6, 8], [11]))

        if crop_image:
            self.add(pr.ControlMap(CropImageFromMask(image_size=image_size,
                                                     crop_size=crop_size),
                                   [10, 11, 0, 4], [12, 13, 11, 14]))

        self.add(pr.ControlMap(CreateScoremaps(image_size=image_size,
                                               crop_size=crop_size,
                                               variance=sigma,
                                               crop_image=crop_image),
                               [11, 10], [15]))
        self.add(pr.SequenceWrapper({0:{}}))
        




























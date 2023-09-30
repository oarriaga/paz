import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.pipelines.detection import PreprocessBoxes
from pose import EfficientPosePreprocess
from processors import (MatchTransformations, TransformRotation,
                        ConcatenateTransformation, ConcatenateScale)


class AugmentPose(SequentialProcessor):
    def __init__(self, model, split=pr.TRAIN, num_classes=8, size=512,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[1.0, 1.0, 1.0, 1.0], num_pose_dims=3):
        super(AugmentPose, self).__init__()
        self.preprocess_image = EfficientPosePreprocess(model)

        # box processors
        self.scale_boxes = pr.ScaleBox()
        args = (num_classes, model.prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # transformation processor
        self.preprocess_transformation = MatchTransformations(
            model.prior_boxes, num_pose_dims)

        self.concat_transformation = ConcatenateTransformation()
        self.concat_scale = ConcatenateScale()

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes', 'rotation',
                                      'translation_raw', 'class']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0, 1, 2]))
        self.add(pr.ControlMap(self.scale_boxes, [3, 1], [3], keep={1: 1}))
        self.add(pr.ControlMap(self.preprocess_boxes, [4], [5], keep={4: 4}))
        self.add(pr.ControlMap(TransformRotation(num_pose_dims), [3], [3]))
        self.add(pr.ControlMap(self.preprocess_transformation,
                               [4, 3], [3], keep={4: 4}))
        self.add(pr.ControlMap(self.preprocess_transformation, [4, 5], [8]))
        self.add(pr.ControlMap(self.concat_transformation, [3, 6], [8]))
        self.add(pr.ControlMap(self.concat_scale, [5, 1], [8]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {2: {'boxes': [len(model.prior_boxes), 4 + num_classes]},
             4: {'transformation': [len(model.prior_boxes),
                                    3 * num_pose_dims + 2]}}))
